#%%
# !pip install colorspacious --break-system-packages
!pip uninstall colour --break-system-packages
!pip install colour-science --break-system-packages

#%%
import colour
import numpy as np


def colour_stripe(S=1, samples=360):
    H = np.linspace(0, 1, samples)

    HSV = colour.utilities.tstack([H, np.ones(samples) * S, np.ones(samples)])
    RGB = colour.HSV_to_RGB(HSV)

    return RGB[np.newaxis, ...]


RGB = np.resize(colour_stripe(), [36, 360, 3])

colour.plotting.plot_image(colour.cctf_encoding(RGB * 0.5));

CAM16 = colour.convert(RGB, 'RGB', 'CAM16')
CAM16_UL = colour.CAM_Specification_CAM16(
    np.full(CAM16.J.shape, 0.5), CAM16.C, CAM16.h)

RGB_PU = colour.convert(CAM16_UL, 'CAM16', 'RGB')

colour.plotting.plot_image(colour.cctf_encoding(RGB_PU));

#%%
def colour_swatches(n_colors=10, S=1.0):
    # Generate evenly spaced hues
    H = np.linspace(0, 1, n_colors, endpoint=False)
    
    # Create HSV values for each swatch
    HSV = colour.utilities.tstack([H, np.ones(n_colors) * S, np.ones(n_colors)])
    RGB = colour.HSV_to_RGB(HSV)
    
    # Create square swatches by repeating colors
    swatch_size = 50
    RGB_swatches = np.zeros((swatch_size, n_colors * swatch_size, 3))
    for i in range(n_colors):
        RGB_swatches[:, i*swatch_size:(i+1)*swatch_size] = RGB[i]
    
    return RGB_swatches

# Plot original RGB swatches
RGB_swatches = colour_swatches(n_colors=10)
colour.plotting.plot_image(colour.cctf_encoding(RGB_swatches * 0.5))

# Convert to CAM16 and adjust lightness
CAM16 = colour.convert(RGB_swatches, 'RGB', 'CAM16')
CAM16_UL = colour.CAM_Specification_CAM16(
    np.full(CAM16.J.shape, 0.5), CAM16.C, CAM16.h)

# Convert back to RGB and plot
RGB_PU = colour.convert(CAM16_UL, 'CAM16', 'RGB')
colour.plotting.plot_image(colour.cctf_encoding(RGB_PU))

#%%
def perceptually_uniform_colour_swatches(n_colors=10, S=1.0):
    # Generate evenly spaced hues in CAM16
    h = np.linspace(0, 360, n_colors, endpoint=False)
    
    # Create CAM16 specification with constant lightness and chroma
    J = np.full(n_colors, 50) # Lightness of 50
    C = np.full(n_colors, 50) # Chroma of 50
    CAM16_spec = colour.CAM_Specification_CAM16(J, C, h)
    
    # Convert to RGB
    RGB = colour.convert(CAM16_spec, 'CAM16', 'RGB')
    
    # Create square swatches by repeating colors
    swatch_size = 50
    RGB_swatches = np.zeros((swatch_size, n_colors * swatch_size, 3))
    for i in range(n_colors):
        RGB_swatches[:, i*swatch_size:(i+1)*swatch_size] = RGB[i]
    
    return RGB_swatches

# Plot perceptually uniform swatches
RGB_swatches_PU = perceptually_uniform_colour_swatches(n_colors=10)
colour.plotting.plot_image(colour.cctf_encoding(RGB_swatches_PU))


