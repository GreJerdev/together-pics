import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import json


class PyTorchResNetClassifier:
    """
    A class that uses PyTorch and ResNet18 to classify images.
    Supports both pre-trained ImageNet classification and custom fine-tuning.
    """
    
    def __init__(self, num_classes=1000, pretrained=True, device=None):
        """
        Initialize the ResNet18 classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
            pretrained (bool): Whether to use pre-trained weights (default: True)
            device (str): Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer for custom number of classes
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load ImageNet class labels if using pre-trained model
        if pretrained and num_classes == 1000:
            self.class_labels = self._load_imagenet_labels()
        else:
            self.class_labels = None
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels."""
        # This is a simplified version - in practice, you'd load from a file
        # For now, we'll return a basic mapping
        return {i: f"Class_{i}" for i in range(1000)}
    
    def preprocess_cv2_image(self, cv2_image):
        """
        Preprocess a CV2 image (numpy array) for classification.
        
        Args:
            cv2_image (numpy.ndarray): CV2 image array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Apply transforms
            image_tensor = self.transform(pil_image).unsqueeze(0)
            
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing CV2 image: {e}")
            return None
    
    def classify_image(self, image_input, top_k=5):
        """
        Classify an image and return top-k predictions.
        
        Args:
            image_input (str or torch.Tensor): Image path or preprocessed tensor
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (class_id, confidence, class_name)
        """
        # Preprocess if input is a path
        if isinstance(image_input, str):
            image_tensor = self.preprocess_image(image_input)
        else:
            image_tensor = image_input
            
        if image_tensor is None:
            return []
        
        with torch.no_grad():
            # Get predictions
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                class_id = top_indices[i].item()
                confidence = top_probs[i].item()
                class_name = self.class_labels.get(class_id, f"Class_{class_id}") if self.class_labels else f"Class_{class_id}"
                results.append((class_id, confidence, class_name))
            
            return results
    
    def classify_batch(self, image_paths, top_k=5):
        """
        Classify a batch of images.
        
        Args:
            image_paths (list): List of image file paths
            top_k (int): Number of top predictions to return per image
            
        Returns:
            dict: Dictionary mapping image paths to their predictions
        """
        results = {}
        
        for image_path in image_paths:
            try:
                predictions = self.classify_image(image_path, top_k)
                results[image_path] = predictions
            except Exception as e:
                print(f"Error classifying {image_path}: {e}")
                results[image_path] = []
        
        return results
    
    def classify_folder(self, folder_path, extensions=None, top_k=5):
        """
        Classify all images in a folder.
        
        Args:
            folder_path (str): Path to the folder containing images
            extensions (list): List of file extensions to include (default: common image formats)
            top_k (int): Number of top predictions to return per image
            
        Returns:
            dict: Dictionary mapping image paths to their predictions
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(folder_path).glob(f"*{ext}"))
            image_paths.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        # Convert to strings
        image_paths = [str(path) for path in image_paths]
        
        return self.classify_batch(image_paths, top_k)
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_labels': self.class_labels
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.class_labels = checkpoint.get('class_labels', None)
        print(f"Model loaded from {filepath}")
    
    def set_class_labels(self, labels):
        """
        Set custom class labels.
        
        Args:
            labels (dict): Dictionary mapping class IDs to class names
        """
        self.class_labels = labels
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
            dict: Model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet18',
            'num_classes': self.num_classes,
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'pretrained': hasattr(self.model, 'pretrained') and self.model.pretrained
        }


# Example usage and testing functions
def test_classifier():
    """Test the PyTorchResNetClassifier with sample images."""
    # Initialize classifier
    classifier = PyTorchResNetClassifier()
    
    # Print model info
    info = classifier.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with sample images if they exist
    sample_images = [
        "download.jpg",
        "images.jpg", 
        "images (1).jpg"
    ]
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            print(f"\nClassifying {image_path}:")
            predictions = classifier.classify_image(image_path, top_k=3)
            for i, (class_id, confidence, class_name) in enumerate(predictions):
                print(f"  {i+1}. {class_name} (ID: {class_id}) - {confidence:.4f}")


if __name__ == "__main__":
    test_classifier()
