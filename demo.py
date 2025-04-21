import torch
from realesrgan import RealESRGAN
from PIL import Image
import numpy as np
import cv2

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the RealESRGAN model with scale 4 (general model for multiple image types)
sr_model = RealESRGAN(device, scale=4)

# Load the specific model weights (realesr-general-x4v3.pth)
sr_model.load_weights('D:/alpr code/realesr-general-x4v3.pth')  # Correct path to your model

# Function to enhance an image using Real-ESRGAN
def enhance_with_esrgan(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))  # Convert to PIL
    enhanced_img = sr_model.predict(pil_img)  # Enhance image
    enhanced_cv2 = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV
    return enhanced_cv2

# Example usage
image_path = "D:/alpr code/output/demo.jpg"
img = cv2.imread(image_path)

# Enhance the image
enhanced_img = enhance_with_esrgan(img)

# Display enhanced image
cv2.imshow('Enhanced Image', enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
