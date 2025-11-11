import os
from imagekitio import ImageKit
from dotenv import load_dotenv

load_dotenv()

def upload_to_imagekit(file_path):
    try:
        private_key = os.getenv('IMAGEKIT_PRIVATE_KEY')
        public_key = os.getenv('IMAGEKIT_PUBLIC_KEY')
        url_endpoint = os.getenv('IMAGEKIT_URL_ENDPOINT')
        
        if not all([private_key, public_key, url_endpoint]):
            raise Exception("Missing ImageKit credentials in environment variables")
        
        imagekit = ImageKit(
            private_key=private_key,
            public_key=public_key,
            url_endpoint=url_endpoint
        )
        
        if not imagekit:
            raise Exception("Failed to initialize ImageKit client")
        
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as file:
            upload_result = imagekit.upload_file(
                file=file,
                file_name=filename
            )

        print(f"Upload result: {upload_result}")
        print(f"Upload result type: {type(upload_result)}")

        if not upload_result:
            raise Exception("Upload failed - no result returned")
        if isinstance(upload_result, dict):
            if upload_result.get("response") and upload_result["response"].get("url"):
                url = upload_result["response"]["url"]
            else:
                url = None
        elif hasattr(upload_result, 'url'):
            url = upload_result.url
        elif isinstance(upload_result, dict):
            url = upload_result.get("url")
        else:
            raise Exception(f"Unexpected response format: {type(upload_result)}")
            
        if not url:
            raise Exception(f"No URL found in response. Response: {upload_result}")

        print(f"Successfully got URL: {url}")
        return url

    except Exception as e:
        print(f"ImageKit upload error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None


if __name__ == "__main__":
    file_path = "./Generated_charts/real_data_20250827_122645.png"
    url = upload_to_imagekit(file_path)
    if url:
        print(f"Uploaded image URL: {url}")

