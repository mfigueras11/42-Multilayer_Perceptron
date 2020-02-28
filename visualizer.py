import turtle
import numpy as np
from time import sleep

def setup():
    window = turtle.Screen()
    turtle.tracer(False)
    turtle.bgcolor(1., .41, .38)
    turtle.colormode(255)

def draw_circle(x, y, r, color=(155, 155, 155), fill=True, caption=None):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    if fill:
        turtle.begin_fill()
    turtle.pensize(2)
    turtle.fillcolor(*color)
    turtle.pencolor(255, 255, 255)
    turtle.circle(r)

    if fill:
        turtle.end_fill()

    if caption:
        turtle.penup()
        turtle.goto(x, y + r+ 3)
        turtle.pendown()
        turtle.pencolor(244, 129, 64)
        turtle.write(str(caption), align='center')


    
def draw_layer(l, x, r, y =0, color=(155, 155, 155), spacing=10, fill=True):
    a = len(l) * (2*r + spacing)
    y = y - r -spacing//2  + a // 2
    norm = (l - l.min()) / (l.max() - l.min())
    for i, n in enumerate(l):
        draw_circle(x, y, r, color=(int(254.*norm[i]), int(254.*norm[i]), int(254.*norm[i])), fill=fill, caption=f"{n:.2f}")
        y -= (spacing + 2*r)

def draw_network(n, r, color=(155, 155, 155), x = 0, spacing=80, fill=True):
    a = (len(n)-1) * (2*r + spacing)
    x = x + r + a//2
    for l in n[::-1]:
        draw_layer(np.mean(l, axis=0), x, r, color=color, fill=fill)
        x -= (spacing + 2*r)
