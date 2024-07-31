import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Dinov2Config, Dinov2Model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
image_height, image_width = 1080, 1920

checkpoint = "facebook/dinov2-base"

model_config = Dinov2Config.from_pretrained(checkpoint, image_size=(image_height, image_width))
model = Dinov2ForImageClassification(model_config) 
or 
model = AutoModel(model_config) 
image_processor = AutoImageProcessor.from_pretrained(
    checkpoint, image_size={"height": image_height, "width": image_width}
)
"""

class SemanticRichEncoder:
    def __init__(self, checkpoint='facebook/dinov2-base', image_size = 518):
        self.checkpoint = checkpoint
        self.config = Dinov2Config.from_pretrained(checkpoint, image_size = image_size)
        self.model = AutoModel.from_config(self.config)
        #self.model = Dinov2Model(self.config)

        # Init BitImageProcessor 
        # The image processor used by dinov2 is defined by the class 'transformers.BitImageProcessor'
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, size={"shortest_edge": image_size}, do_center_crop = False)
        
        
    def extract_features(self, image):
        if not isinstance(image, Image.Image):
            #raise ValueError("The input must be a PIL Image object.")
            pil_image = Image.fromarray(image) #expects RGB image
        
        
        inputs = self.processor(images=image, return_tensors="pt")#.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        
        #print(last_hidden_states.shape)
    
        batch_size, seq_length, hidden_size = last_hidden_states.shape
        
        patch_dim = int((seq_length - 1) ** 0.5)

        class_token = last_hidden_states[:, 0, :]
        patch_embeddings = last_hidden_states[:, 1:, :]

        feature_map = patch_embeddings.reshape(batch_size, patch_dim, patch_dim, hidden_size).permute(0, 3, 1, 2)
    
        return feature_map, class_token
        
    def visualize_feature_map(self, feature_map=None, image=None, num_channels=6, cmap='viridis'):
        if (feature_map is not None and image is not None) or (feature_map is None and image is None):
            raise ValueError("Either feature_map or image must be provided, but not both.")
        
        if feature_map is None:
            feature_map,_ = self.extract_features(image)
        
        # Select the first image in the batch and the first few channels
        feature_map = feature_map[0, :num_channels, :, :].cpu()
        
        # Plot the selected channels
        fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 2, 2))
        for i in range(num_channels):
            ax = axes[i]
            ax.imshow(feature_map[i], cmap=cmap)
            ax.axis('off')
            ax.set_title(f'Channel {i+1}')
        plt.tight_layout()
        plt.show()

    def visualize_pca_feature_map(self, feature_map=None, image=None, cmap='viridis', n_components=3, **kwargs):
        if (feature_map is not None and image is not None) or (feature_map is None and image is None):
            raise ValueError("Either feature_map or image must be provided, but not both.")
        
        if feature_map is None:
            feature_map, _ = self.extract_features(image)
        
        batch_size = feature_map.shape[0]
        hidden_size = feature_map.shape[1]
        patch_dim = feature_map.shape[2]
        
        # Reshape feature map to [batch_size, patch_dim * patch_dim, hidden_size]
        feature_map = feature_map.view(batch_size, hidden_size, -1).permute(0, 2, 1)
        
        # Perform PCA on the first image in the batch
        features = feature_map[0].cpu().numpy()  # Shape: [patch_dim * patch_dim, hidden_size]
        
        pca = PCA(n_components=n_components, **kwargs)
        pca_features = pca.fit_transform(features)  # Shape: [patch_dim * patch_dim, n_components]
        
        # Normalize the PCA features for visualization
        pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0)) #if (pca_features.max(axis=0) -pca_features.min(axis=0)) != 0 else None
        
        # Reshape to the original patch grid
        pca_features = pca_features.reshape(patch_dim, patch_dim, n_components)
        
        # Plot the PCA components
        fig, axes = plt.subplots(1, n_components, figsize=(n_components * 2, 2))
        for i in range(n_components):
            ax = axes[i]
            ax.imshow(pca_features[:, :, i], cmap=cmap)
            ax.axis('off')
            ax.set_title(f'PCA Component {i+1}')
        
        plt.tight_layout()
        plt.show()
