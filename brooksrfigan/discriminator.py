import tensorflow as tf

class ConvNet_default(tf.keras.Model):
    def __init__(self, input_shape, kernel_size=3, filters=32):
        super(ConvNet_default, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.mp1 = tf.keras.layers.MaxPool2D()

        self.conv2 = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.mp2 = tf.keras.layers.MaxPool2D()

        self.conv3 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=kernel_size, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()        
        self.classifier = tf.keras.layers.Dense(1,activation='sigmoid')
    
    def call(self, inputs, training: bool):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.flatten(x)
        return self.classifier(x)

