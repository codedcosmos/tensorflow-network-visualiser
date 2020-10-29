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


import cairo, math
from PIL import Image
import sys
import numpy as np
import cv2


#     _   __     __                      __      _    ___                  ___
#    / | / /__  / /__      ______  _____/ /__   | |  / (_)______  ______ _/ (_)_______  _____
#   /  |/ / _ \/ __/ | /| / / __ \/ ___/ //_/   | | / / / ___/ / / / __ `/ / / ___/ _ \/ ___/
#  / /|  /  __/ /_ | |/ |/ / /_/ / /  / ,<      | |/ / (__  ) /_/ / /_/ / / (__  )  __/ /
# /_/ |_/\___/\__/ |__/|__/\____/_/  /_/|_|     |___/_/____/\__,_/\__,_/_/_/____/\___/_/
#


#########################
# Configuration
#########################


class VisualisationConfig:
	def __init__(self, width, height):
		# Image
		self.width = width
		self.height = height

		# Buffers
		self.border_buffer = min(width, height) * 0.09
		self.layer_buffer = min(width, height) * 0.03

		# Weights
		self.draw_weights = False
		self.weight_thickness = 2
		self.minimum_weight_brightness = 0.5

		# Determines how many neurons to draw before
		# Compacting them since theres to many
		# Use 0 to disable this feature
		# Layers with less than n neurons will draw like normal
		self.max_neurons_normal_draw = 0

		# Sets the size of dense neurons
		# 1 means they will be packed without any gaps
		# 0.5 means the gap is the size of the neuron
		self.neuron_gap = 0.8

		# Enables/Disables rgb colour
		self.rgb_colour = False

	# Render data
	def calculate_render_data(self, input_layer, model):
		# Calculate drawing values
		num_layers = len(model.layers) + 1

		self.canvas_width = self.width - self.border_buffer * 2
		self.canvas_height = self.height - self.border_buffer * 2

		self.layer_width = (self.canvas_width - self.layer_buffer * (num_layers - 1)) / num_layers
		self.layer_height = self.canvas_height

		# Input shape
		self.input_shape = input_layer.shape

		# Obvious checks
		if self.canvas_width < 0 or self.canvas_height < 0:
			raise Exception(
				"Invalid canvas width or canvas height, Must be at least 1. The border_buffer size is likely to large")

		if self.layer_width < 0 or self.layer_height < 0:
			raise Exception(
				"Invalid layer width or layer height, Must be at least 1. The layer_buffer size is likely to large")

		# Print details
		print("Calculated image properties")
		print("Width:", self.width, "Height:", self.height)
		print("Canvas Width:", round(self.canvas_width, 1), "Canvas Height:", round(self.canvas_height, 1))
		print("Layer Width:", round(self.layer_width, 1), "Layer Height:", round(self.layer_height, 1))

	# Smart getters
	def should_draw_normal_dense(self, num_neurons):
		# Check if feature is disabled
		if (self.max_neurons_normal_draw == 0):
			return True

		return num_neurons < self.max_neurons_normal_draw

	def get_input_size_dense(self):
		return self.input_shape[1]

	# Setters
	def set_weight_thickness(self, thickness):
		self.weight_thickness = thickness

	def set_minimum_weight_brightness(self, brightness):
		self.minimum_weight_brightness = brightness

	def set_border_buffer(self, border_buffer):
		self.border_buffer = border_buffer

	def set_layer_buffer(self, layer_buffer):
		self.layer_buffer = layer_buffer

	def enable_draw_weights(self):
		self.draw_weights = True

	def disable_draw_weights(self):
		self.draw_weights = False

	def set_max_neurons_for_normal_draw(self, n):
		self.max_neurons_normal_draw = n

	def set_neuron_gap(self, gap_size):
		self.neuron_gap = gap_size

	def enable_rgb_colour(self):
		self.rgb_colour = True

	def disable_rgb_colour(self):
		self.rgb_colour = False

	# Getters
	def get_resolution(self):
		return (self.width, self.height)

	def get_buffers(self):
		return (self.border_buffer, self.layer_buffer)

	def get_canvas_details(self):
		return (self.layer_width, self.layer_height, self.canvas_width, self.canvas_height)

	def get_neuron_gap(self):
		return self.neuron_gap

	def get_max_neurons_normal_draw(self):
		return self.max_neurons_normal_draw

	def should_draw_weights(self):
		return self.draw_weights

	def get_weight_thickness(self):
		return self.weight_thickness

	def get_minimum_weight_brightness(self):
		return self.minimum_weight_brightness

	def is_rgb_colour_enabled(self):
		return self.rgb_colour


#########################
# Utility
#########################


def make_even_and_round(num):
	num = round(num)
	if num % 2:
		return num + 1
	else:
		return num


#########################
# Drawing Primitives
#########################


def draw_background(cr, rgb, width, height):
	cr.set_source_rgb(rgb[0], rgb[1], rgb[2])
	cr.rectangle(0, 0, width, height)
	cr.fill()


def draw_rect(cr, rgb, x, y, w, h):
	cr.set_source_rgb(rgb[0], rgb[1], rgb[2])
	cr.rectangle(x, y, w, h)
	cr.fill()


def draw_circle(cr, rgb, x, y, radius):
	cr.set_source_rgb(rgb[0], rgb[1], rgb[2])
	cr.arc(x, y, radius, 0, 2 * math.pi)
	cr.fill()


def draw_h_centered_text(cr, rgb, x, y, font_size, text):
	cr.set_source_rgb(rgb[0], rgb[1], rgb[2])
	cr.select_font_face("Purisa", cairo.FONT_SLANT_NORMAL,
						cairo.FONT_WEIGHT_NORMAL)
	cr.set_font_size(font_size)

	# Get extents
	(tx, ty, tw, th, dx, dy) = cr.text_extents(text)

	cr.move_to(x - tw / 2, y)
	cr.show_text(text)


def draw_text(cr, rgb, x, y, font_size, text):
	cr.set_source_rgb(rgb[0], rgb[1], rgb[2])

	cr.select_font_face("Purisa", cairo.FONT_SLANT_NORMAL,
						cairo.FONT_WEIGHT_NORMAL)
	cr.set_font_size(font_size)

	cr.move_to(x, y)
	cr.show_text(text)


def draw_line(cr, rgb, x1, y1, x2, y2, thickness):
	cr.set_source_rgb(rgb[0], rgb[1], rgb[2])
	cr.set_line_width(thickness)

	cr.move_to(x1, y1)
	cr.line_to(x2, y2)
	cr.stroke()


#########################
# Drawing Complex
#########################


def draw_neuron(cr, inner_rgb, outer_rgb, x, y, inner_radius, outer_radius):
	draw_circle(cr, outer_rgb, x, y, outer_radius)
	draw_circle(cr, inner_rgb, x, y, inner_radius)

def draw_dense_layer(cr, visconfig, layer_num, weights):
	# Extract
	width, height = visconfig.get_resolution()
	border_buffer, layer_buffer = visconfig.get_buffers()
	layer_width, layer_height, canvas_width, canvas_height = visconfig.get_canvas_details()

	biases = weights[1]
	weights = weights[0]

	# Prepare new "last_weight_positions"
	new_last_weight_positions = []

	# x is easy
	x = border_buffer + (layer_buffer + layer_width) * layer_num + layer_width / 2
	# Neuron size
	num_neurons = len(biases)
	# Draw height
	draw_height = height - border_buffer * 2

	# Draw normal or draw to many neurons case
	if visconfig.should_draw_normal_dense(num_neurons):
		# Calculate y
		y_gap = draw_height / num_neurons
		outer_size = y_gap / 2 * visconfig.get_neuron_gap()
		inner_size = outer_size * 0.8

		for i, neuron in enumerate(biases):
			y = border_buffer + y_gap * i + y_gap / 2

			draw_neuron(cr, get_rgb(visconfig, neuron), (1, 1, 1), x, y, inner_size, outer_size)
			new_last_weight_positions.append((i, x, y))
	else:
		# Neurons to draw count per side
		draw_neuron_count = visconfig.get_max_neurons_normal_draw() / 2
		draw_neuron_count = make_even_and_round(draw_neuron_count)

		# Calculate y
		y_gap = draw_height / (draw_neuron_count * 2 + 3)
		outer_size = y_gap / 2 * visconfig.get_neuron_gap()
		inner_size = outer_size * 0.8

		# Draw top
		for i in range(0, draw_neuron_count):
			y = border_buffer + (y_gap * i) + (y_gap / 2)

			draw_neuron(cr, (0, 0, 0), (1, 1, 1), x, y, inner_size, outer_size)
			new_last_weight_positions.append((i, x, y))

		# Draw Text
		y = border_buffer + y_gap * (draw_neuron_count + 1.25) + (y_gap / 2)
		draw_h_centered_text(cr, (1, 1, 1), x - outer_size, y, outer_size * 3, "(" + str(num_neurons) + ")")

		# Draw bot
		for i in range(draw_neuron_count + 3, draw_neuron_count * 2 + 3):
			y = border_buffer + (y_gap * i) + (y_gap / 2)

			draw_neuron(cr, (0, 0, 0), (1, 1, 1), x, y, inner_size, outer_size)
			new_last_weight_positions.append((i, x, y))

	return new_last_weight_positions


def draw_dense_layer_weights(cr, visconfig, layer_num, weights, last_weights_positions):
	def draw_weights(neuron_index, nx, ny):
		# Gets position by index
		def get_pos_by_index(index):
			for i, x, y in last_weights_positions:
				if i == index:
					return (x, y)

			return None

		# If there aren't any previous ones skip
		if len(last_weights_positions) == 0:
			return

		for index, weight_group in enumerate(weights):
			# Get position by index
			position = get_pos_by_index(index)
			if position == None:
				continue

			# Extract
			x, y = position

			# Get weight
			weight = weight_group[neuron_index] \
					* (1 - visconfig.get_minimum_weight_brightness()) \
					+ visconfig.get_minimum_weight_brightness()

			# Draw line
			draw_line(cr, get_rgb(visconfig, weight), x, y, nx, ny, visconfig.get_weight_thickness())

	# Extract
	width, height = visconfig.get_resolution()
	border_buffer, layer_buffer = visconfig.get_buffers()
	layer_width, layer_height, canvas_width, canvas_height = visconfig.get_canvas_details()

	biases = weights[1]
	weights = weights[0]

	# Prepare new "last_weight_positions"
	new_last_weight_positions = []

	# x is easy
	x = border_buffer + (layer_buffer + layer_width) * layer_num + layer_width / 2
	# Neuron size
	num_neurons = len(biases)
	# Draw height
	draw_height = height - border_buffer * 2

	# Draw normal or draw to many neurons case
	if visconfig.should_draw_normal_dense(num_neurons):
		# Calculate y
		y_gap = draw_height / num_neurons

		for i, neuron in enumerate(biases):
			y = border_buffer + y_gap * i + y_gap / 2

			draw_weights(i, x, y)
			new_last_weight_positions.append((i, x, y))
	else:
		# Neurons to draw count per side
		draw_neuron_count = visconfig.get_max_neurons_normal_draw() / 2
		draw_neuron_count = make_even_and_round(draw_neuron_count)

		# Calculate y
		y_gap = draw_height / (draw_neuron_count * 2 + 3)

		# Draw top
		for i in range(0, draw_neuron_count):
			y = border_buffer + (y_gap * i) + (y_gap / 2)

			draw_weights(i, x, y)
			new_last_weight_positions.append((i, x, y))

		# Draw bot
		for i in range(draw_neuron_count + 3, draw_neuron_count * 2 + 3):
			y = border_buffer + (y_gap * i) + (y_gap / 2)

			draw_weights(i, x, y)
			new_last_weight_positions.append((i, x, y))

	return new_last_weight_positions


def draw_conv2D_layer(cr, visconfig, layer_num, weights):
	# Extract
	width, height = visconfig.get_resolution()
	border_buffer, layer_buffer = visconfig.get_buffers()
	layer_width, layer_height, canvas_width, canvas_height = visconfig.get_canvas_details()

	# Calculate Kernel details
	weights = weights[0]
	kernel_width = weights.shape[0]
	kernel_height = weights.shape[1]
	num_kernels = weights.shape[3]

	num_cells_on_x_axis = 2 + kernel_width
	num_cells_on_y_axis = 2 + (kernel_height-1) + (kernel_height*num_kernels)

	# Calculate sizes
	max_cell_width = layer_width/num_cells_on_x_axis
	max_cell_height = layer_height/num_cells_on_y_axis
	cell_size = min(max_cell_width, max_cell_height)
	inner_cell_gap = cell_size * 0.1
	inner_cell_size = cell_size * 0.9 - inner_cell_gap*2

	# x is easy
	x = border_buffer + (layer_buffer + layer_width) * layer_num + (layer_width-(cell_size*kernel_width))/2
	y_gap = (layer_height - (2 + num_kernels - 1) * cell_size) / num_kernels + cell_size

	# Draw
	for i in range(0, num_kernels):
		# Calculate y
		y = border_buffer + (y_gap * i) + y_gap/4

		# Draw border
		draw_rect(cr, (1, 1, 1), x, y, kernel_width * cell_size, kernel_height * cell_size)

		for xi in range(0, kernel_width):
			for yi in range(0, kernel_height):
				inner_x = x + (inner_cell_gap/2+cell_size)*xi + inner_cell_gap
				inner_y = y + (inner_cell_gap/2+cell_size)*yi + inner_cell_gap

				neuron = weights[xi, yi, 0, i]

				draw_rect(cr, (neuron, neuron, neuron), inner_x, inner_y, inner_cell_size, inner_cell_size)


def draw_flatten_layer(cr, visconfig, layer_num):
	# Extract
	border_buffer, layer_buffer = visconfig.get_buffers()
	layer_width, layer_height, canvas_width, canvas_height = visconfig.get_canvas_details()

	# x
	x = border_buffer + (layer_buffer + layer_width) * layer_num + (layer_width/2)

	# Draw
	draw_line(cr, (1, 1, 1), x, border_buffer, x, border_buffer+layer_height, visconfig.get_weight_thickness())

def draw_input_layer_as_dense(cr, visconfig):
	size = visconfig.get_input_size_dense()
	weights = [[], ([0] * size)]

	return draw_dense_layer(cr, visconfig, 0, weights)


#########################
# Calculate
#########################


def normalise_frames(frames):
	# Gets first item in a mess of list, tuples and numpy arrays
	def get_first(item):
		if type(item).__module__ == np.__name__:
			return item.flat[0]
		elif type(item) is list or type(item) is tuple:
			if len(item) == 0:
				return []

			return get_first(item[0])
		else:
			return item

	# Recursively get min/max
	def get_by_func(function, item, current):
		if type(item).__module__ == np.__name__ and len(item.shape) > 0:
			for sub_item in np.nditer(item):
				sub_val = get_by_func(function, sub_item, current)
				current = function(sub_val, current)
			return current

		elif type(item) is list or type(item) is tuple:
			for sub_item in item:
				sub_val = get_by_func(function, sub_item, current)
				current = function(sub_val, current)
			return current

		return function(item, current)

	# Normalises values by min/max
	# Recursively get min/max
	def normalise(item, min_value, max_value):
		def normalise_singular_value(item):
			return (item - min_value) / (max_value - min_value)

		if type(item).__module__ == np.__name__ and len(item.shape) > 0:
			with np.nditer(item, op_flags=['readwrite']) as it:
				for x in it:
					x[...] = normalise_singular_value(x)
			return item

		elif type(item) is list or type(item) is tuple:
			items = []
			for sub_item in item:
				items.append(normalise(sub_item, min_value, max_value))
			return items

		return normalise_singular_value(item)

	# Min and max values for each frame
	min_in_frames = []
	max_in_frames = []

	# Get min/max
	for frame in frames:
		# Extract
		samples_seen, layer_data = frame

		for layer in layer_data:
			# Extract
			layer_num, layer_name, weights = layer

			# Get first
			first = get_first(weights)

			# Get min/max
			min_in_frames.append(get_by_func(min, weights, first))
			max_in_frames.append(get_by_func(max, weights, first))

	min_value = min(min_in_frames)
	max_value = max(max_in_frames)

	print("Received min/max values of ", str(min_value) + ", " + str(max_value))

	# Normalise
	normalised_frames = []
	for frame in frames:
		# Extract
		samples_seen, layer_data = frame

		normalised_layers = []

		for layer in layer_data:
			# Extract
			layer_num, layer_name, weights = layer

			normalised_weights = normalise(weights, min_value, max_value)
			normalised_layers.append((layer_num, layer_name, normalised_weights))

		normalised_frames.append((samples_seen, normalised_layers))

	return normalised_frames


def calculate_frame(model, samples_seen):
	layer_data = []

	for layer_num, layer in enumerate(model.layers):
		layer_name = layer.__class__.__name__
		weights = layer.get_weights()

		layer_data.append((layer_num, layer_name, weights))

	return (samples_seen, layer_data)


def get_rgb(visconfig, c):
	if visconfig.is_rgb_colour_enabled():
		return generate_coloured_rgb(c)
	return (c, c, c)


def generate_coloured_rgb(c):
	r = 0
	b = 0
	g = 0

	# Check red
	if c > 0 and c < 0.5:
		r = 1.0 - pow(2.0, 4.0) * pow(c - 0.25, 2.0)

	# Check green
	if c > 0.25 and c < 0.75:
		g = 1.0 - pow(2.0, 4.0) * pow(c - 0.50, 2.0)

	# Check blue
	if c > 0.5 and c < 0.75:
		b = 1.0 - pow(2.0, 4.0) * pow(c - 0.75, 2.0)

	# WhiteMax
	if c > 0.75 and c < 1:
		r = g = 1.0 - pow(2.0, 4.0) * pow(c - 1.0, 2.0)
		b = 1.0 - 0.1 * pow(c - 1.0, 2.0)

	return (r, g, b)


#########################
# Generative Stages
#########################


def render_frame(visconfig, frame):
	# Extract
	width, height = visconfig.get_resolution()
	border_buffer, layer_buffer = visconfig.get_buffers()
	samples_seen, layer_data = frame

	# Prepare rendering context
	ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
	cr = cairo.Context(ims)

	# Draw Background
	draw_background(cr, (0.1, 0.1, 0.1), width, height)

	# Draw input (layer)
	last_weights_positions = draw_input_layer_as_dense(cr, visconfig)

	# Draw weights
	for layer_num, layer_name, weights in layer_data:
		if layer_name == "Dense":
			# Draw weights if needed
			if visconfig.should_draw_weights():
				last_weights_positions = draw_dense_layer_weights(cr, visconfig, layer_num + 1, weights, last_weights_positions)
		elif layer_name == "Conv2D":
			last_weights_positions = []
		elif layer_name == "Flatten":
			last_weights_positions = []

	# Draw layers
	for layer_num, layer_name, weights in layer_data:
		if layer_name == "Dense":
			draw_dense_layer(cr, visconfig, layer_num + 1, weights)
		elif layer_name == "Conv2D":
			draw_conv2D_layer(cr, visconfig, layer_num + 1, weights)
		elif layer_name == "Flatten":
			draw_flatten_layer(cr, visconfig, layer_num + 1)
		else:
			print("Incompatible layer type:", layer_name)

	text_gap = border_buffer/5
	draw_text(cr, (1, 1, 1), text_gap, height-text_gap, border_buffer/2, "Samples Seen " + str(samples_seen))

	return ims


def render_frames(input_layer, model, frames, visconfig):
	# Extract
	width, height = visconfig.get_resolution()

	# Calculate drawing values
	visconfig.calculate_render_data(input_layer, model)

	# Prepare
	drawn_frames = []

	# Render each frame
	for i, frame in enumerate(frames):
		# Console output
		sys.stdout.write("\rRendering frame %d/%d" % (i+1, len(frames)))
		sys.stdout.flush()

		# Render frame
		rendered_frame = render_frame(visconfig, frame)

		# Convert frame into image for PIL
		buf = rendered_frame.get_data()
		array = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
		im = Image.fromarray(array, 'RGBA')

		# Save
		drawn_frames.append(im)

	# Prevent text being on same line
	print("")
	sys.stdout.flush()

	return drawn_frames


#########################
# Generative
#########################


def render_to_gif(input_layer, model, frames, visconfig, file_name):
	# Calculate drawing values
	visconfig.calculate_render_data(input_layer, model)

	# Prepare
	drawn_frames = render_frames(input_layer, model, frames, visconfig)

	# Render frames into gif
	first_frame = drawn_frames[0]
	drawn_frames.pop(0)

	first_frame.save(file_name, save_all=True, append_images=drawn_frames, fps=6)
	print("Saved", file_name)


def render_to_avi(input_layer, model, frames, visconfig, file_name):
	# Extract
	width, height = visconfig.get_resolution()

	# Prepare
	drawn_frames = render_frames(input_layer, model, frames, visconfig)

	# Render frames into avi
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter(file_name, fourcc, 6, (width, height))

	for frame in drawn_frames:
		rgbframe = frame.convert('RGB')
		open_cv_image = np.array(rgbframe)
		open_cv_image = open_cv_image[:, :, ::-1].copy()

		video.write(open_cv_image)

	video.release()
	print("Saved", file_name)


def render_to_png(input_layer, model, frame, visconfig, file_name):
	# Extract
	width, height = visconfig.get_resolution()

	# Calculate drawing values
	visconfig.calculate_render_data(input_layer, model)

	# Render frame
	rendered_frame = render_frame(visconfig, frame)

	# Convert frame into image for PIL
	buf = rendered_frame.get_data()
	array = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
	im = Image.fromarray(array, 'RGBA')

	# Save
	im.save(file_name)
	print("Saved", file_name)
