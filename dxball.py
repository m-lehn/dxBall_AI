import pygame
import os
from PIL import Image

# Class for blocks
class Block(pygame.sprite.Sprite):
    def __init__(self, posx, posy, size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface(size)
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.topleft = posx, posy

# Class for paddle/plate
class Plate(pygame.sprite.Sprite):
    def __init__(self, width, bottom):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width, 14])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.rect.left = 100
        self.rect.bottom = bottom

    def moveLeft(self):
        if self.rect.left - 20 >= 0:
            self.rect.move_ip(-20, 0)

    def moveRight(self):
        if self.rect.right + 20 <= self.area.right:
            self.rect.move_ip(20, 0)

# Class for ball
class Ball(pygame.sprite.Sprite):
    def __init__(self, plate):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 255, 255), (10, 10), 10)
        self.rect = self.image.get_rect()
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.spX, self.spY = -4, 4
        self.onPlate = 0
        self.plate = plate
        # Set the initial position of the ball - often starting on or just above the paddle
        self.rect.centerx = self.plate.rect.centerx
        self.rect.bottom = self.plate.rect.top - 1  # Position the ball just above the paddle


    def update(self):
        # Horizontal bounce
        if self.rect.left + self.spX < 0 or self.rect.right + self.spX > self.area.right:
            self.spX *= -1

        # Vertical bounce
        if self.rect.top + self.spY < 0:
            self.spY *= -1

        if self.rect.colliderect(self.plate.rect):
            # Determine where the ball hits the paddle
            hit_position = (self.rect.centerx - self.plate.rect.left) / self.plate.rect.width

            # Reverse the vertical speed
            self.spY *= -1

            # Adjust the horizontal speed based on hit position
            if hit_position < 0.2:  # Extreme left
                self.spX -= 3  # Increase leftward movement
            elif hit_position < 0.4:  # Moderate left
                self.spX -= 1  # Slightly increase leftward movement
            elif hit_position > 0.8:  # Extreme right
                self.spX += 3  # Increase rightward movement
            elif hit_position > 0.6:  # Moderate right
                self.spX += 1  # Slightly increase rightward movement
            # Center region (0.4 to 0.6) makes no change to horizontal speed

            # Ensure the ball moves upwards
            self.spY = -abs(self.spY)

            self.onPlate = 0  # Reset contact flag

        # Update position
        self.rect.move_ip(self.spX, self.spY)

# Game class to manage the game state
class Game:
    def __init__(self, window_size, frame_directory, block_width, block_height, block_gap, block_rows, block_cols):
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("DX-Ball")
        self.clock = pygame.time.Clock()
        self.running = True

        self.frame_directory = frame_directory
        os.makedirs(self.frame_directory, exist_ok=True)  # Ensure the directory exists
        self.frame_count = 0
        # Check for already existing jpg files
        self.initialize_frame_count()

        self.block_width = block_width
        self.block_height = block_height
        self.block_gap = block_gap
        self.block_rows = block_rows
        self.block_cols = block_cols

        self.init_game()

    # Check for existing jpg files and set frame_count appropriately
    def initialize_frame_count(self):
        # List all files in the directory
        files = os.listdir(self.frame_directory)
        jpg_files = [f for f in files if f.endswith('.png')]

        if jpg_files:
            # Extract numbers from filenames and find the maximum
            numbers = [int(f.split('.')[0]) for f in jpg_files]  # Assumes files are named as 'number.jpg'
            self.frame_count = max(numbers)  # Set frame count to the highest existing file number

    def create_blocks(self):
        blocks = pygame.sprite.Group()
        block_width, block_height = self.block_width, self.block_height
        num_blocks_x = self.block_cols  # Number of blocks horizontally
        num_blocks_y = self.block_rows   # Number of blocks vertically
        gap = self.block_gap  # Space between blocks

        # Calculate total blocks width to center horizontally
        total_blocks_width = num_blocks_x * (block_width + gap) - gap  # Subtract one gap to fit within screen
        start_x = (self.screen.get_width() - total_blocks_width) // 2

        # Calculate total blocks height to start from a certain point vertically
        total_blocks_height = num_blocks_y * (block_height + gap) - gap  # Subtract one gap to maintain vertical alignment
        start_y = (self.screen.get_height() // 4) - (total_blocks_height // 2)  # Start at one-fourth from the top

        # Create blocks grid
        for x in range(num_blocks_x):
            for y in range(num_blocks_y):
                pos_x = start_x + x * (block_width + gap)  # Include gap in position
                pos_y = start_y + y * (block_height + gap)  # Include gap in position
                block = Block(pos_x, pos_y, (block_width, block_height))
                blocks.add(block)
        return blocks

    def init_game(self):
        # Initialize game objects
        current_size = self.screen.get_size()
        self.plate = Plate(current_size[0] // 4, current_size[1] - 30)
        self.ball = Ball(self.plate)
        self.blocks = self.create_blocks()
        self.allsprites = pygame.sprite.RenderPlain((self.ball, self.plate))

    def save_frame(self):
        self.frame_count += 1
        frame_filename = os.path.join(self.frame_directory, f"{self.frame_count:06d}.png")
        # Get the current screen size
        current_size = self.screen.get_size()
        # Reduce the width and height
        new_size = (current_size[0] // 5, current_size[1] // 5)
        # Resize the surface
        resized_surface = pygame.transform.scale(self.screen, new_size)
        # Convert Pygame surface to a string buffer (RGB format)
        surface_data = pygame.image.tostring(resized_surface, "RGB")
        # Create a Pillow image from the string buffer
        img = Image.frombytes("RGB", new_size, surface_data)
        # Convert to grayscale
        img = img.convert("1")  # Converts the image to binary
        # Save the processed image
        img.save(frame_filename, format="PNG")

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.save_frame()  # Call to save each frame
            self.clock.tick(60)

    def handle_events(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.plate.moveLeft()
        if keys[pygame.K_RIGHT]:
            self.plate.moveRight()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def handle_block_collision(self, block):
        # Calculate overlap in both dimensions
        horizontal_overlap = min(self.ball.rect.right, block.rect.right) - max(self.ball.rect.left, block.rect.left)
        vertical_overlap = min(self.ball.rect.bottom, block.rect.bottom) - max(self.ball.rect.top, block.rect.top)

        # Determine which overlap is greater
        if horizontal_overlap < vertical_overlap:
            # Horizontal collision is more significant
            self.ball.spX *= -1
        else:
            # Vertical collision is more significant
            self.ball.spY *= -1

    def update(self):
        self.allsprites.update()

        # Check for losing the ball
        if self.ball.rect.top > self.screen.get_height():
            self.reset_game()

        # Check for collision with blocks
        blocks_hit = pygame.sprite.spritecollide(self.ball, self.blocks, False)
        if blocks_hit:
            block = blocks_hit[0]  # Assume collision with the first block in the list
            self.handle_block_collision(block)  # Handle the collision with the block
            self.blocks.remove(block)  # Remove the block that was hit

        # Check if the player wins
        if not self.blocks:
            self.reset_game()

        # Display the game win message
        if not self.blocks:
            print("You win!")
            self.running = False

    def reset_game(self):
        print("Resetting game...")
        self.init_game()  # Reinitialize game objects

    def render(self):
        self.screen.fill((0, 0, 0))
        self.allsprites.draw(self.screen)
        self.blocks.draw(self.screen)
        pygame.display.flip()


if __name__ == "__main__":
    window_size = (250, 250)
    frame_directory = "game_frames"
    block_width, block_height = 60, 30
    block_gap = 20
    block_rows, block_cols = 2, 3
    game = Game(window_size, frame_directory, block_width, block_height, block_gap, block_rows, block_cols)
    game.run()
