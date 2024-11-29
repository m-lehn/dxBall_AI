from conv_lstm_model import ConvLSTMNetwork
from conv_lstm_model import ConvLSTMCell
from conv_lstm_model import ConvLSTM

from transformer_model import Head
from transformer_model import MultiHeadAttention
from transformer_model import Block
from transformer_model import FeedForward
from transformer_model import TransformerEncoder
from transformer_model import FramePredictor

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import pygame
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Argument parsing
parser = argparse.ArgumentParser(description="Run DxBall with a specific model.")
parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file (e.g., dxball_model.pt).")

args = parser.parse_args()

# Load the model and switch it to the correct device
model = torch.load(args.model_path)
model = model.to(device)
model.eval()

# Transform for new frame to tensor and normalization
transform = transforms.Compose([
    transforms.Grayscale(),        # Ensure it's grayscale
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

def pil_to_surface(image):
    """Convert a PIL image to a Pygame surface."""
    return pygame.image.fromstring(
        image.tobytes(), image.size, image.mode
    )

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((250, 250))

# Create a clock object
clock = pygame.time.Clock()

# Load and process images
image_dir = 'game_frames'
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[:10]
sequence = torch.stack([transform(Image.open(os.path.join(image_dir, f)).convert('L')) for f in image_files])
sequence = sequence.to(device)

# Process sequence and display loop
with torch.no_grad():
    while True:
        input_tensor = sequence.unsqueeze(0)  # Add batch dimension
        output = model(input_tensor[:, -10:, :, :])  # Use the last 10 frames
        next_frame = output[-1]  # Get the latest frame output from the model

        # Binarize the model's output with a threshold of 0.5
        next_frame = output[-1]  # Get the latest frame output from the model
        next_frame = (next_frame > 0.5).float()  # Convert to binary values (0 or 1)

        # Do not downscale from upscaled images, use the model's output directly
        # Transform the model's output to PIL for displaying
        display_frame = transforms.functional.to_pil_image(next_frame.cpu())
        display_frame = display_frame.convert('RGB')  # Convert grayscale to RGB
        display_frame = display_frame.resize((250, 250), Image.NEAREST)

        # Convert PIL image to Pygame surface for display
        surface = pil_to_surface(display_frame)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Prepare the next frame output by model to be part of the next input sequence
        # Ensuring we do not use the transformed display image for the model input
        next_frame_ready = transform(to_pil_image(next_frame.cpu())).to(device)
        sequence = torch.cat((sequence[1:], next_frame_ready.unsqueeze(0)), 0)  # Shift sequence, drop oldest

        # Limit the framerate
        clock.tick(2)  # adjust the framerate limit

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
