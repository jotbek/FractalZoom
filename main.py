import pyglet
from pyglet.gl import *
import numpy as np
import ctypes
import time   # For timestamping (if needed)
import os     # For file path handling
import tkinter as tk
from tkinter import filedialog

# Initialize tkinter root (it remains hidden)
root = tk.Tk()
root.withdraw()

# --- Shader helper functions ---
def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    source = source.encode('utf-8')
    length = ctypes.c_int(len(source))
    source_ptr = ctypes.cast(ctypes.pointer(ctypes.c_char_p(source)),
                             ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
    glShaderSource(shader, 1, source_ptr, ctypes.byref(length))
    glCompileShader(shader)
    result = ctypes.c_int()
    glGetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(result))
    if not result.value:
        print(f"Error compiling shader type {shader_type}:")
        print_shader_log(shader)
        raise RuntimeError('Shader compilation failed')
    return shader

def print_shader_log(shader):
    length = ctypes.c_int()
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, ctypes.byref(length))
    if length.value > 1:
        log = ctypes.create_string_buffer(length.value)
        glGetShaderInfoLog(shader, length, None, log)
        print(log.value.decode('utf-8'))

def create_program(vertex_source, fragment_source):
    program = glCreateProgram()
    vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_source)
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    result = ctypes.c_int()
    glGetProgramiv(program, GL_LINK_STATUS, ctypes.byref(result))
    if not result.value:
        print("Error linking program:")
        print_program_log(program)
        raise RuntimeError('Program linking failed')
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def print_program_log(program):
    length = ctypes.c_int()
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, ctypes.byref(length))
    if length.value > 1:
        log = ctypes.create_string_buffer(length.value)
        glGetProgramInfoLog(program, length, None, log)
        print(log.value.decode('utf-8'))

# --- Shader sources ---
vertex_shader_source = """
#version 400 core
layout(location = 0) in vec2 position;
out vec2 fragCoord;
void main()
{
    fragCoord = position;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fractal_shaders = {
    'mandelbrot': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz)*6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
    }
    
    void main()
    {
        dvec2 uv = dvec2(fragCoord.xy+1.0)/2.0;
        uv.x *= double(u_resolution.x)/double(u_resolution.y);
        dvec2 c = (uv-dvec2(0.5,0.5))*u_scale + u_center;
        dvec2 z = c;
        int maxIterations = 1000;
        int i;
        for(i = 0; i < maxIterations; i++){
            if(dot(z,z) > 4.0) break;
            z = dvec2(z.x*z.x - z.y*z.y + c.x,
                      2.0*z.x*z.y + c.y);
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'julia': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p-K.xxx, 0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 z = uv;
        dvec2 c = dvec2(-0.8, 0.156);
        int maxIterations = 1000;
        int i;
        for(i=0; i<maxIterations; i++){
            if(dot(z,z)>4.0) break;
            z = dvec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'burning_ship': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p-K.xxx, 0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 c = uv;
        dvec2 z = dvec2(0.0);
        int maxIterations = 1000;
        int i;
        for(i=0;i<maxIterations;i++){
            z = dvec2(abs(z.x), abs(z.y));
            if(dot(z,z)>4.0) break;
            z = dvec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'tricorn': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
        return c.z*mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 c = uv;
        dvec2 z = dvec2(0.0);
        int maxIterations = 1000;
        int i;
        for(i=0;i<maxIterations;i++){
            if(dot(z,z)>4.0) break;
            z = dvec2(z.x*z.x - z.y*z.y, -2.0*z.x*z.y) + c;
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'multibrot3': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
        return c.z*mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 c = uv;
        dvec2 z = dvec2(0.0);
        int maxIterations = 1000;
        int i;
        for(i=0;i<maxIterations;i++){
            if(dot(z,z)>4.0) break;
            double x = z.x;
            double y = z.y;
            z = dvec2(x*x*x - 3.0*x*y*y + c.x,
                      3.0*x*x*y - y*y*y + c.y);
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'multibrot4': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
        return c.z*mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 c = uv;
        dvec2 z = dvec2(0.0);
        int maxIterations = 1000;
        int i;
        for(i=0;i<maxIterations;i++){
            if(dot(z,z)>4.0) break;
            double x = z.x;
            double y = z.y;
            z = dvec2(x*x*x*x - 6.0*x*x*y*y + y*y*y*y + c.x,
                      4.0*x*x*x*y - 4.0*x*y*y*y + c.y);
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'perpendicular_mandelbrot': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
        return c.z*mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 c = uv;
        dvec2 z = dvec2(0.0);
        int maxIterations = 1000;
        int i;
        for(i=0;i<maxIterations;i++){
            if(dot(z,z)>4.0) break;
            z = dvec2(z.x*z.x - z.y*z.y, -2.0*abs(z.x*z.y)) + c;
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """,
    'celtic_mandelbrot': """
    #version 400 core
    in vec2 fragCoord;
    out vec4 outColor;
    uniform dvec2 u_center;
    uniform double u_scale;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform bool u_color_animation;
    vec3 hsv2rgb(vec3 c){
        vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
        vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
        return c.z*mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
    }
    void main(){
        dvec2 uv = (fragCoord.xy)*u_scale + u_center;
        dvec2 c = uv;
        dvec2 z = dvec2(0.0);
        int maxIterations = 1000;
        int i;
        for(i=0;i<maxIterations;i++){
            if(dot(z,z)>4.0) break;
            double x_new = abs(z.x*z.x - z.y*z.y) + c.x;
            z.y = 2.0*z.x*z.y + c.y;
            z.x = x_new;
        }
        float t = float(i)/float(maxIterations);
        float hueShift = u_color_animation ? u_time : 0.0;
        vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
        outColor = vec4(color,1.0);
    }
    """
}

# Shader for drawing the color palette (remains unchanged)
palette_vertex_shader_source = """
#version 400 core
layout(location = 0) in vec2 position;
out float t_value;
void main(){
    t_value = position.x;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

palette_fragment_shader_source = """
#version 400 core
in float t_value;
out vec4 outColor;
uniform float u_time;
vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
    vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
    return c.z*mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
}
void main(){
    float t = t_value;
    float hueShift = u_time;
    vec3 color = hsv2rgb(vec3(mod(t*5.0+hueShift,1.0),1.0,1.0));
    outColor = vec4(color,1.0);
}
"""

# --- Window and OpenGL initialization ---
config = pyglet.gl.Config(double_buffer=True, major_version=4, minor_version=0)
window = pyglet.window.Window(resizable=True, config=config)
window.set_caption("Fractal Viewer - jbk made THIS in 2024!")
glClearColor(0, 0, 0, 1)

# Choose initial fractal and compile shaders
current_fractal = 'mandelbrot'
program = create_program(vertex_shader_source, fractal_shaders[current_fractal])
palette_program = create_program(palette_vertex_shader_source, palette_fragment_shader_source)

# Get uniform locations
def get_uniform_locations():
    global u_center_location, u_scale_location, u_resolution_location
    global u_time_location, u_color_animation_location
    u_center_location = glGetUniformLocation(program, b'u_center')
    u_scale_location = glGetUniformLocation(program, b'u_scale')
    u_resolution_location = glGetUniformLocation(program, b'u_resolution')
    u_time_location = glGetUniformLocation(program, b'u_time')
    u_color_animation_location = glGetUniformLocation(program, b'u_color_animation')

get_uniform_locations()
u_palette_time_location = glGetUniformLocation(palette_program, b'u_time')

# Initial view settings
initial_center = [0.0, 0.0]
initial_scale = 3.0
center = initial_center.copy()
scale = initial_scale

# Mouse/keyboard state variables
zoom_in = False
zoom_out = False
color_animation = False
color_time = 0.0
pan_left = False
pan_right = False
pan_up = False
pan_down = False
show_help = False
confirm_exit = False

# For the F1 Help hover hint
show_hint = False

# Vertices for a full-screen quad
vertices = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0
], dtype=np.float32)

# Setup VAO and VBO for fractal rendering
def setup_vao():
    global vao, vbo
    vao = GLuint()
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao)
    vbo = GLuint()
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

setup_vao()

# Setup VAO and VBO for the color palette
palette_vao = GLuint()
glGenVertexArrays(1, palette_vao)
glBindVertexArray(palette_vao)
palette_vertices = np.array([
    0.0, -1.0,
    1.0, -1.0,
    0.0,  1.0,
    0.0,  1.0,
    1.0, -1.0,
    1.0,  1.0
], dtype=np.float32)
palette_vbo = GLuint()
glGenBuffers(1, palette_vbo)
glBindBuffer(GL_ARRAY_BUFFER, palette_vbo)
glBufferData(GL_ARRAY_BUFFER, palette_vertices.nbytes,
             palette_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
             GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

# Mouse position
mouse_x = 0
mouse_y = 0

# Create zoom level label
zoom_label = pyglet.text.Label('Zoom: 1.00x',
                               font_name='Arial',
                               font_size=12,
                               bold=True,
                               x=10, y=window.height-10,
                               anchor_x='left', anchor_y='top',
                               color=(255,255,255,255))

# Create FPS display
fps_display = pyglet.window.FPSDisplay(window)
fps_display.label.font_size = 12
fps_display.label.color = (255,255,255,255)
fps_display.label.bold = True
fps_display.label.anchor_x = 'right'
fps_display.label.anchor_y = 'top'

# Create F1 Help label in bottom-right corner
help_label = pyglet.text.Label('F1 Help',
                               font_name='Arial',
                               font_size=12,
                               bold=True,
                               x=window.width-10, y=10,
                               anchor_x='right', anchor_y='bottom',
                               color=(255,255,255,255))

# Help entries (including screenshot key)
help_entries = [
    ('Left Mouse Button', 'Zoom In'),
    ('Right Mouse Button', 'Zoom Out'),
    ('Arrow Keys', 'Pan View'),
    ('Backspace', 'Toggle Color Animation'),
    ('Enter', 'Reset View'),
    ('Escape', 'Exit Application'),
    ('F1', 'Toggle Help Screen'),
    ('F5', 'Mandelbrot Set'),
    ('F6', 'Julia Set'),
    ('F7', 'Burning Ship'),
    ('F8', 'Tricorn'),
    ('F9', 'Multibrot (Power 3)'),
    ('F10', 'Multibrot (Power 4)'),
    ('F11', 'Perpendicular Mandelbrot'),
    ('F12', 'Celtic Mandelbrot'),
    ('S', 'Save Screenshot')
]

def draw_text_with_outline(label, outline_color=(0,0,0,255), outline_offset=1):
    x = label.x
    y = label.y
    original_color = label.color
    label.color = outline_color
    offsets = [(-outline_offset, -outline_offset),
               (-outline_offset, 0),
               (-outline_offset, outline_offset),
               (0, -outline_offset),
               (0, outline_offset),
               (outline_offset, -outline_offset),
               (outline_offset, 0),
               (outline_offset, outline_offset)]
    for dx, dy in offsets:
        label.x = x+dx
        label.y = y+dy
        label.draw()
    label.x = x
    label.y = y
    label.color = original_color
    label.draw()

def update(dt):
    global center, scale, color_time
    color_time += dt * 0.1
    if confirm_exit or show_help:
        return
    # Pan view with arrow keys
    pan_speed = scale * 0.02
    if pan_left:
        center[0] -= pan_speed
    if pan_right:
        center[0] += pan_speed
    if pan_up:
        center[1] += pan_speed
    if pan_down:
        center[1] -= pan_speed
    if zoom_in or zoom_out:
        width, height = window.get_size()
        ndc_x = (mouse_x / width)
        ndc_y = (mouse_y / height)
        ndc_y = 1.0 - ndc_y
        factor = 0.98 if zoom_in else 1.02 if zoom_out else 1.0
        new_scale = scale * factor
        if new_scale > initial_scale:
            new_scale = initial_scale
            factor = new_scale / scale
        c_re = (ndc_x - 0.5)*scale + center[0]
        c_im = (ndc_y - 0.5)*scale + center[1]
        scale = new_scale
        center[0] = c_re - (ndc_x - 0.5)*scale
        center[1] = c_im - (ndc_y - 0.5)*scale

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT)
    width, height = window.get_size()
    # Draw fractal
    glUseProgram(program)
    glUniform2d(u_center_location, center[0], center[1])
    glUniform1d(u_scale_location, scale)
    glUniform2f(u_resolution_location, width, height)
    glUniform1f(u_time_location, color_time)
    glUniform1i(u_color_animation_location, int(color_animation))
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindVertexArray(0)
    glUseProgram(0)
    # Draw palette
    glUseProgram(palette_program)
    glUniform1f(u_palette_time_location, color_time)
    glBindVertexArray(palette_vao)
    glViewport(0,0,200,20)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glViewport(0,0,width,height)
    glBindVertexArray(0)
    glUseProgram(0)
    # Update and draw zoom label
    zoom_factor = initial_scale / scale
    zoom_label.text = f'Zoom: {zoom_factor:.2f}x'
    draw_text_with_outline(zoom_label)
    # Update and draw FPS display
    fps_display.label.x = width-10
    fps_display.label.y = height-10
    draw_text_with_outline(fps_display.label)
    # Draw F1 Help label
    help_label.x = width-10
    help_label.y = 10
    draw_text_with_outline(help_label)
    # Draw hover hint box if mouse is over F1 Help label
    if show_hint:
        lines = [f"{key}: {desc}" for key, desc in help_entries]
        hint_text = "\n".join(lines)
        padding = 5
        max_line_width = max([pyglet.text.Label(line, font_name='Arial', font_size=12).content_width for line in lines])
        line_height = pyglet.text.Label("A", font_name='Arial', font_size=12).content_height
        box_width = max_line_width + 2*padding
        box_height = len(lines)*line_height + 2*padding
        hint_x = width - 10 - box_width
        hint_y = 10 + help_label.content_height + 10
        hint_box = pyglet.shapes.Rectangle(hint_x, hint_y, box_width, box_height, color=(50,50,50))
        hint_box.opacity = 200
        hint_box.draw()
        hint_label = pyglet.text.Label(hint_text,
                                       font_name='Arial',
                                       font_size=12,
                                       x=hint_x + padding,
                                       y=hint_y + box_height - padding,
                                       anchor_x='left', anchor_y='top',
                                       multiline=True,
                                       width=box_width - 2*padding,
                                       color=(255,255,255,255))
        hint_label.draw()
    # Draw help or exit overlay if active
    if show_help or confirm_exit:
        overlay.width = width
        overlay.height = height
        overlay.draw()
        if show_help:
            font_size = 14
            font_name = 'Courier New'
            line_height = font_size * 1.5
            start_y = height//2 + (len(help_entries)//2)*line_height
            key_color = (200,200,200,255)
            desc_color = (255,255,255,255)
            max_key_width = 0
            for key, _ in help_entries:
                label = pyglet.text.Label(f'{key}', font_name=font_name, font_size=font_size, bold=False)
                if label.content_width > max_key_width:
                    max_key_width = label.content_width
            for i, (key, desc) in enumerate(help_entries):
                y = start_y - i*line_height
                key_label = pyglet.text.Label(f'{key}', font_name=font_name, font_size=font_size,
                                              x=width*0.1, y=y, anchor_x='left', anchor_y='center', color=key_color)
                key_label.draw()
                desc_label = pyglet.text.Label(desc, font_name=font_name, font_size=font_size,
                                               x=width*0.1+max_key_width+10, y=y, anchor_x='left', anchor_y='center', color=desc_color)
                desc_label.draw()
        elif confirm_exit:
            exit_text = "Are you sure you want to exit the program?\nPress Escape again to exit."
            exit_label = pyglet.text.Label(exit_text, font_name='Arial', font_size=16,
                                           x=width//2, y=height//2, anchor_x='center', anchor_y='center',
                                           multiline=True, width=width*0.8, color=(255,255,255,255))
            exit_label.draw()

@window.event
def on_resize(width, height):
    glViewport(0,0,width,height)
    zoom_label.y = height - 10
    help_label.x = width - 10
    help_label.y = 10
    overlay.width = width
    overlay.height = height

@window.event
def on_mouse_press(x, y, button, modifiers):
    global zoom_in, zoom_out, mouse_x, mouse_y
    if confirm_exit or show_help:
        return
    mouse_x, mouse_y = x, y
    if button == pyglet.window.mouse.LEFT:
        zoom_in = True
    elif button == pyglet.window.mouse.RIGHT:
        zoom_out = True

@window.event
def on_mouse_release(x, y, button, modifiers):
    global zoom_in, zoom_out
    if confirm_exit or show_help:
        return
    if button == pyglet.window.mouse.LEFT:
        zoom_in = False
    elif button == pyglet.window.mouse.RIGHT:
        zoom_out = False

@window.event
def on_mouse_motion(x, y, dx, dy):
    global mouse_x, mouse_y, show_hint
    if confirm_exit or show_help:
        return
    mouse_x, mouse_y = x, y
    left_bound = window.width - 10 - help_label.content_width
    right_bound = window.width - 10
    bottom_bound = 10
    top_bound = 10 + help_label.content_height
    if left_bound <= x <= right_bound and bottom_bound <= y <= top_bound:
        show_hint = True
    else:
        show_hint = False

@window.event
def on_key_press(symbol, modifiers):
    global center, scale, color_animation, show_help, confirm_exit
    global pan_left, pan_right, pan_up, pan_down, program, current_fractal
    if confirm_exit:
        if symbol == pyglet.window.key.ESCAPE:
            window.close()
        else:
            confirm_exit = False
        return True
    if show_help:
        if symbol == pyglet.window.key.F1 or symbol == pyglet.window.key.ESCAPE:
            show_help = False
        return True
    if symbol == pyglet.window.key.ESCAPE:
        confirm_exit = True
    elif symbol == pyglet.window.key.F1:
        show_help = True
    elif symbol == pyglet.window.key.ENTER:
        center = initial_center.copy()
        scale = initial_scale
    elif symbol == pyglet.window.key.BACKSPACE:
        color_animation = not color_animation
    elif symbol == pyglet.window.key.LEFT:
        pan_left = True
    elif symbol == pyglet.window.key.RIGHT:
        pan_right = True
    elif symbol == pyglet.window.key.UP:
        pan_up = True
    elif symbol == pyglet.window.key.DOWN:
        pan_down = True
    elif symbol == pyglet.window.key.S:
        filename = filedialog.asksaveasfilename(title="Save Screenshot",
                                                  defaultextension=".png",
                                                  filetypes=[("PNG files", "*.png")])
        if filename:
            screenshot = pyglet.image.get_buffer_manager().get_color_buffer()
            screenshot.save(filename)
            print(f"Screenshot saved as {filename}")
    elif symbol in [pyglet.window.key.F5, pyglet.window.key.F6, pyglet.window.key.F7,
                    pyglet.window.key.F8, pyglet.window.key.F9, pyglet.window.key.F10,
                    pyglet.window.key.F11, pyglet.window.key.F12]:
        fractal_keys = {
            pyglet.window.key.F5: 'mandelbrot',
            pyglet.window.key.F6: 'julia',
            pyglet.window.key.F7: 'burning_ship',
            pyglet.window.key.F8: 'tricorn',
            pyglet.window.key.F9: 'multibrot3',
            pyglet.window.key.F10: 'multibrot4',
            pyglet.window.key.F11: 'perpendicular_mandelbrot',
            pyglet.window.key.F12: 'celtic_mandelbrot'
        }
        current_fractal = fractal_keys[symbol]
        glDeleteProgram(program)
        program = create_program(vertex_shader_source, fractal_shaders[current_fractal])
        get_uniform_locations()
        center = initial_center.copy()
        scale = initial_scale
        setup_vao()
    return True

@window.event
def on_key_release(symbol, modifiers):
    global pan_left, pan_right, pan_up, pan_down
    if symbol == pyglet.window.key.LEFT:
        pan_left = False
    elif symbol == pyglet.window.key.RIGHT:
        pan_right = False
    elif symbol == pyglet.window.key.UP:
        pan_up = False
    elif symbol == pyglet.window.key.DOWN:
        pan_down = False
    return True

# Create overlay rectangle for help/exit screens
overlay = pyglet.shapes.Rectangle(0, 0, window.width, window.height, color=(0,0,0))
overlay.opacity = int(0.8*255)

# Schedule update function at 60 FPS
pyglet.clock.schedule_interval(update, 1/60)

# Run the application
pyglet.app.run()
