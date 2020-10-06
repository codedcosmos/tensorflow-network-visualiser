# Tensorflow Network Visualiser by codedcosmos
#
# Tensorflow Network Visualiser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License 3 as published by
# the Free Software Foundation.
# Tensorflow Network Visualiser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License 3 for more details.
# You should have received a copy of the GNU General Public License 3
# along with Tensorflow Network Visualiser.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf
import numpy as np
import network_visualiser

# Configurable
BATCH_SIZE = 1
EPOCHS = 1
STEP_SIZE = 1000

# Load the images from mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Combine training and testing data into a single dataset
# This is typically not recommended since it is useful to verify if a neural network works well or has overfitted
# But a highly accurate network isn't the goal of this project
x_train = np.concatenate((x_train, x_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

# Get the number of images
print("Loaded", len(x_train), "images")
print("With shape", x_train.shape)

# Normalise images and add an extra axis for tensorflow
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

# Process
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Input shape
input_layer = tf.keras.layers.Input(shape=(28, 28, 1))

# Create model
model = tf.keras.Sequential([
    input_layer,
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(3, (5, 5), activation='relu'),
    tf.keras.layers.Conv2D(2, (2, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(10),
])

# Model functions
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Visualisation config
visconfig = network_visualiser.VisualisationConfig(1920, 1080)
visconfig.set_border_buffer(100)
visconfig.set_layer_buffer(30)
visconfig.set_max_neurons_for_normal_draw(18)
visconfig.set_neuron_gap(0.7)
visconfig.enable_draw_weights()

def custom_train(epochs=1):
    frames = []

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every STEP_SIZE batches.
            if step % STEP_SIZE == 0:
                print(
                    "Training - Step: %d/%d - Loss: %.4f"
                    % (step, len(train_dataset), float(loss_value))
                )

                # Append frame
                frames.append(network_visualiser.calculate_frame(model, step))

    # Normalise
    frames = network_visualiser.normalise_frames(frames)

    # Draw gif
    network_visualiser.render_to_gif(input_layer, model, frames, visconfig, "example_conv.gif")

custom_train()