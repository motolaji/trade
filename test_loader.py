from image_loader import ImageLoader

def test_loader():
    # Load a test image
    image_path = '../trademark/L3D_dataset/images/fffff6ec-af49-4d8e-98df-b7eb0f53067a.jpg'  # Add your test image
    try:
        image_data = ImageLoader.load_image(image_path)
        print("Image loaded successfully!")
        print(f"Image shape: {image_data['numpy'].shape}")
    except Exception as e:
        print(f"Error loading image: {str(e)}")

if __name__ == "__main__":
    test_loader()