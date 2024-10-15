class Config:
    def __init__(self):
        #self.dropout = 0.1
        self.learning_rate = 0.0001
        self.weight_decay = 1e-5
        self.num_epochs = 30
        self.dataset_dir = "C:/Users/sasir/Desktop/DSL-BIO/data"
        self.results_dir = 'Results' 
        self.model_save_dir = 'saved_models'

    def __str__(self):
        return f"Dropout: {self.dropout} \
            \nLearning rate: {self.learning_rate} \
            \nWeight decay: {self.weight_decay} \
            \nEpochs: {self.num_epochs}"
    

class ConfigG:
    def __init__(self):
        # Dataset settings
        self.dataset_dir = "C:/Users/sasir/Desktop/DSL-BIO/data"  # Directory with 'train', 'val', 'test' folders
        self.model_save_dir = 'saved_models'  # Directory to save the trained model
        
        # Training settings
        self.batch_size = 16
        self.num_epochs = 10
        self.learning_rate = 0.0001
        self.weight_decay = 1e-4
        
        # Model settings
        self.model_name = 'googlenet'  # You can add logic to select other models if needed
    
        
        # Input image size for Inception v3 (GoogLeNet)
        self.img_size = 299