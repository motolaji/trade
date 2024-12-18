from image_loader import ImageLoader

def test_loader():
    # Load a test image
    image_path = './L3D-dataset/dataset/images/2dbcd05e-b951-42bc-8361-eb3034aebc6d.JPG'  # Add your test image
    try:
        image_data = ImageLoader.load_image(image_path)
        print("Image loaded successfully!")
        print(f"Image shape: {image_data['numpy'].shape}")
    except Exception as e:
        print(f"Error loading image: {str(e)}")

if __name__ == "__main__":
    test_loader()

 