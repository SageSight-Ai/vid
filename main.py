import os
import requests
import shutil
import random
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from datetime import datetime
from io import BytesIO
import boto3
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
import math
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import validators

# API Keys and Credentials
authorization_header = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IlJmUnEwV0FmTEhuV2RobkZXMWllXyJ9.eyJpc3MiOiJodHRwczovL3JlY3JhZnRhaS51cy5hdXRoMC5jb20vIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMTI3MjY0MjcwMjUyNjUwNjEzMjAiLCJhdWQiOlsicmVjcmFmdC1iYWNrZW5kIiwiaHR0cHM6Ly9yZWNyYWZ0YWkudXMuYXV0aDAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTcxNTgyMzg1MSwiZXhwIjoxNzE1OTEwMjUxLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIHJlYWQ6YWxsIG9mZmxpbmVfYWNjZXNzIiwiYXpwIjoiNWhnejBlMm9RcG8yQTlYcDhhODByalZ3QkVId0hLbEIifQ.iAu0_LmIPC4dGm646c1VaHwcREWP0xO4KQwF2VVNVdObWQv3orJhd39Z8R-5ZdIwG91jHS4C7kXPq8bZCM1udOOwDKJJqCo6_wDgyWHGcEeOfEskLNYmsFil2YVOu0popDDUqoX52R6o5yPIbbpO8qJV7ElqNVGUQNronZC0bOsUVLq8smy2I9DMtD0rJAcBl5hZzO-3LMMQwpe3KzylwhZJVeNngLvG6oNb3RfBMzF_sHBcOOFvFFZ09vta1XUV6Fbbp2-hrr8HKRgy3JxpFVFmcX9kcNDmTnf2_XKtetZkgXUVNPpjcJg9ZNbEEiqIXc5Jaqg5LCA29YnVo-XiTQ"
authorization_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiOTYxNDA5YzctY2IyMS00ZDJhLWJiZWUtMTdmZTM3MWEyOGNmIiwidHlwZSI6ImFwaV90b2tlbiJ9.ugKWyxuOM7_db4O1jHbhskk37GAf302roHEKR2qi-cQ"
aws_access_key_id = "AKIAZQ3DN6O2PREMNYIL"
aws_secret_access_key = "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"
s3_bucket_name = "gen.videos.s3"

# Function to generate content
def generate_content(user_input):
    url = "https://api.edenai.run/v2/text/chat"
    payload = {
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 7302,
        "providers": "perplexityai",
        "settings": "{ \"perplexityai\": \"pplx-7b-online\" }",
        "text": f"{user_input}",
        "chatbot_global_action": "Think Like a TikTok Star: Imagine you're holding your phone, ready to film. You press record... what happens next? What do you say? What do you do to grab your viewers' attention and keep them hooked? Dialogue/Actions Only: Write the script as if someone is speaking directly to the camera. Only include what the person will say and any actions they might take on-screen. Do not include any of the following: Title Outlining Camera Directions (e.g., 'cut to close-up') Music Suggestions Hashtags Special Effects Additional Instructions Dialogue: What I say directly to the camera. Actions: Things I do on camera to make the video more engaging (e.g., 'Hold up a spoon', 'Point to something off-screen', 'Smile and wink'). Make sure the script follows these TikTok rules: The first 3 Seconds are KEY: Grab attention IMMEDIATELY! Keep it Short & Sweet: Under 60 seconds, around 400 characters max. Simple Language: Talk like you're talking to a friend! Tell Them What to Do: End with a call to action (Like, comment, follow, etc.) Script Requirements: Showtime! Write the script exactly as you would speak it while filming. Dialogue: What you will say directly to the camera. Actions: Describe any physical actions that add visual interest: The TikTok Formula for Success: Hook, Line, and Sinker: The first 3 seconds are EVERYTHING! Your opening must grab attention instantly and stop viewers from scrolling. Short & Sweet: The entire script should be deliverable in under 60 seconds and contain a maximum of 400 characters. TikTok is all about quick, digestible content! Speak Like a Friend: Use simple, everyday language that everyone can understand. Avoid jargon and technical terms. Get Them Involved: End with a clear call to action that encourages viewers to engage:here is the idea or the topic:"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {authorization_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['perplexityai']['generated_text']
        return content
    else:
        print("Failed to generate content. Response status:", response.status_code)
        return None

# Function to generate prompts from content
def generate_prompts(content):
    url = "https://api.edenai.run/v2/text/chat"
    payload = {
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 7302,
        "providers": "perplexityai",
        "settings": "{ \"perplexityai\": \"pplx-7b-online\" }",
        "text": content,
        "chatbot_global_action": "You are a creative art director for cinematic footage. You are young but a genius in your field. Your vision is modern aesthetic[aesthetic: a particular theory or conception of beauty or art : a particular taste for or approach to what is pleasing to the senses and especially sight. modernist aesthetics.], and the techniques and shooting angles you use are atypical. The company you work for is a media company that creates videos on social media. They hired you because of your stunning aesthetic and atypical visuals and FX. After AI took over, you decided to work from home. By reading the content they send to you, then writing a perfect image scene prompts.[[Modern vibes]]… [scenes should be fitting the content sequence]  Each prompt should be in a line. The structure of an AI art prompt[[image content/subject, description of action, state, and mood],[ Make at least 25 image prompts ]. The images should be creative , with some specials FX, and filters, engaging and  detailed prompts, start writing the prompts directly, [also try to avoid anything that may cuz a text in the image, because the Ai model they using is not able to process the text in the scenes] ((no empty lines)) [Make 25 image prompt at least], HERE IS THE CONTENT :"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {authorization_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        prompts = response_data['perplexityai']['generated_text']
        return prompts
    else:
        print("Failed to generate prompts. Response status:", response.status_code)
        return None

# Function to generate a random seed
def generate_random_seed():
    return random.randint(1, 10000000)

# Function to remove empty lines from prompts
def remove_empty_lines(prompts):
    return '\n'.join(filter(lambda x: x.strip(), prompts.split('\n')))

# Function to download audio with unique filename
def download_audio(audio_url):
    unique_filename = f"audio_{uuid.uuid4()}.mp3"  # Generate unique filename
    response = requests.get(audio_url, stream=True)
    if response.status_code == 200:
        with open(unique_filename, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        print(f"Audio downloaded successfully as {unique_filename}")
        return unique_filename
    else:
        print("Failed to download audio. Response status:", response.status_code)
        return None

# Function to convert text to speech
def convert_text_to_speech(text):
    url = "https://api.edenai.run/v2/audio/text_to_speech"
    payload = {
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_original_response": True,
        "settings": "{ \"openai\": \"en_echo\" }",
        "rate": 0,
        "pitch": 0,
        "volume": 0,
        "sampling_rate": 0,
        "providers": "openai",
        "text": text
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {authorization_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        audio_url = response_data['openai']['audio_resource_url']
        print("Text converted to speech successfully.")
        return audio_url
    else:
        print("Failed to convert text to speech. Response status:", response.status_code)
        return None

# Function to add fade-in and fade-out transition to each clip
def add_transitions(clip_list):
    transition_duration = 1  # Duration of fade-in and fade-out transitions in seconds
    transition_clips = [clip.fadein(transition_duration).fadeout(transition_duration) for clip in clip_list]
    return transition_clips

# Function to apply zoom effect to each image clip
def apply_zoom_effect(image_clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size
        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]
        new_size = [(size + (size % 2)) for size in new_size]
        img = img.resize(new_size, Image.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)
        img = img.crop([x, y, new_size[0] - x, new_size[1] - y]).resize(base_size, Image.LANCZOS)
        result = np.array(img)
        img.close()
        return result
    return image_clip.fl(effect)

# Function to generate the video and upload to S3
def generate_video(image_urls, audio_url):
    # Download images and create list of image clips
    image_clips = []
    for url in image_urls:
        if not validators.url(url):
            print(f"Invalid image URL: {url}")
            continue  # Skip invalid URLs
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        # Resize image to reduce memory usage
        image = image.resize((1080, 1920))  # Adjust dimensions as needed
        image_array = np.array(image)
        # Convert image array to RGB mode if necessary
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        # Create ImageClip and set duration
        image_clip = ImageClip(image_array).set_duration(4)  # Each image lasts for 4 seconds
        # Apply zoom effect to the image clip
        image_clip = apply_zoom_effect(image_clip)
        # Append to list of image clips
        image_clips.append(image_clip)

    # Add fade-in and fade-out transitions to each clip
    transition_clips = add_transitions(image_clips)

    # Concatenate clips to create the final video
    video_clip = concatenate_videoclips(transition_clips, method="compose")

    # Download audio with unique filename
    audio_filename = download_audio(audio_url)
    if not audio_filename:
        return None  # Handle audio download failure

    # Set the audio duration to match the video duration
    audio_clip = AudioFileClip(audio_filename)
    audio_duration = audio_clip.duration
    video_duration = video_clip.duration
    if audio_duration > video_duration:
        audio_clip = audio_clip.subclip(0, video_duration)
    elif audio_duration < video_duration:
        video_clip = video_clip.subclip(0, audio_duration)

    # Set the audio volume to 0.8
    audio_clip = audio_clip.volumex(0.8)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Generate unique filename for the video based on current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"video_{uuid.uuid4()}.mp4"

    # Write the video file
    video_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac", fps=24)

    # Upload video to AWS S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    try:
        s3_client.upload_file(output_filename, s3_bucket_name, output_filename)
        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{output_filename}"
        print(f"Video uploaded to S3: {s3_url}")
        return s3_url  # Return the S3 URL of the video
    except Exception as e:
        print(f"Error uploading video to S3: {e}")
        return None

# Define headers for the image generation requests
headers = {
    'authority': 'api.recraft.ai',
    'accept': '/',
    'accept-language': 'en-US,en;q=0.8',
    'authorization': authorization_header,
    'content-type': 'application/json',
    'origin': 'https://app.recraft.ai',
    'referer': 'https://app.recraft.ai/',
    'sec-ch-ua': '"Not A(Brand";v="99", "Brave";v="121", "Chromium";v="121"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
}

# FastAPI app
app = FastAPI()

# Function to process a single video generation request
def process_video_request(data):
    user_input = data.get("user_input")
    if not user_input:
        return JSONResponse(status_code=400, content={"error": "Missing user_input"})
    generated_content = generate_content(user_input)
    generated_prompts = generate_prompts(generated_content)
    audio_url = convert_text_to_speech(generated_content)

    # Generate image URLs concurrently with delays and retries
    image_urls = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, prompt in enumerate(remove_empty_lines(generated_prompts).split('\n')):
            prompt = prompt.strip()  # Remove leading/trailing whitespaces
            print(f"Processing prompt: {prompt}")
            # Combine the prompt with the extension
            full_prompt = f"{prompt}"
            # Define the JSON data for the POST request
            json_data = {
                'prompt': full_prompt + "4k, high res, full hd, hyper realistic",
                'image_type': 'digital_illustration',
                'negative_prompt': "Negative prompt placeholder",
                'user_controls': {},
                'layer_size': {
                    'height': 3840,
                    'width': 2160,
                },
                'random_seed': generate_random_seed(),  # Generate random seed
                'style_refs': [],
                'developer_params': {},
            }
            futures.append(executor.submit(generate_image_url_with_retry, headers, json_data))

            # Add delay between image generations within a group
            if (i + 1) % 5 == 0:
                time.sleep(2)  # Adjust delay as needed

        # Add a longer delay between groups of image generations
        time.sleep(10)  # Increased delay for image URL retrieval

        image_urls = [future.result() for future in futures if future.result() is not None]

        # Generate the video and upload to S3
        s3_url = generate_video(image_urls, audio_url)
        if s3_url:
            return JSONResponse(status_code=200, content={"video_url": s3_url})
        else:
            return JSONResponse(status_code=500, content={"error": "Failed to generate or upload video."})

# Function to generate an image URL with retries and extended polling
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_image_url_with_retry(headers, json_data):
    response = requests.post('https://api.recraft.ai/queue_recraft/prompt_to_image', headers=headers, json=json_data)
    # If 'operation_id' is in the JSON body:
    if 'operationId' in response.json():
        operation_id = response.json()['operationId']
        print("operation_id:", operation_id)
        # Poll for the result with extended timeout
        for _ in range(20):  # Adjust attempts as needed
            time.sleep(5)  # Wait 5 seconds between polls
            params = {'operation_id': operation_id}
            response = requests.get('https://api.recraft.ai/poll_recraft', params=params, headers=headers)
            poll_recraft_response = response.json()
            if 'images' in poll_recraft_response:
                break  # Image generation successful, exit loop

        # Process the response if images are available
        if 'images' in poll_recraft_response:
            print("Images:")
            image_id = poll_recraft_response['images'][0]['image_id']
            print("  Image ID:", image_id)
            # Build the URL
            url = f"https://app.recraft.ai/community?imageId={image_id}"
            # Make a GET request to the URL
            html_response = requests.get(url, headers=headers)
            # Check if the request was successful
            if html_response.status_code == 200:
                # Extract image URL from HTML content
                soup = BeautifulSoup(html_response.text, 'html.parser')
                image_url = soup.find("meta", property="og:image")["content"]
                print(image_url)
                return image_url
            else:
                print(f"Failed to retrieve HTML content for image {image_id}. Status code: {html_response.status_code}")

    return None  # Return None to indicate failure

@app.post("/generate_video")
async def generate_video_endpoint(request: Request):
    try:
        data = await request.json()
        return process_video_request(data)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred during video generation."})
