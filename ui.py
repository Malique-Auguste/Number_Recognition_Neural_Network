import pygame as py
import pygame.font

py.init()

class ui_object():
    pos_x = 0
    pos_y = 0
    width = 20
    height = 20
    color = 0
    WHITE = (255, 255, 255)
    root = None
    rect = py.Rect

    def __init__(self, pos_x_, pos_y_, width_, height_, root_):
        self.pos_x = pos_x_
        self.pos_y = pos_y_
        self.width = width_
        self.height = height_
        self.root = root_
        self.rect = py.draw.rect(root_, self.WHITE, (self.pos_x, self.pos_y, self.width, self.height))
        py.display.update()
    
    def clicked(self, mouse_pos):
        if (self.pos_x - (self.width * 0.25)) < mouse_pos[0] and (self.pos_x + (self.width * 1.25)) > mouse_pos[0]:
            if (self.pos_y - (self.height * 0.25)) < mouse_pos[1] and (self.pos_y + (self.height * 1.25)) > mouse_pos[1]:
                return True
        return False

class button(ui_object):
    def __init__(self, pos_x, pos_y, width, height, root):
        ui_object.__init__(self, pos_x, pos_y, width, height, root)

    def clicked(self, mouse_pos):
        if ui_object.clicked(self, mouse_pos):
            self.change_color(1)

    def change_color(self, color_change):
        
        self.color = color_change
        
        if self.color > 1:
            self.color = 1
        elif self.color < 0:
            self.color = 0
        py.draw.rect(self.root, tuple(value * (1 - self.color) for value in self.WHITE), (self.pos_x, self.pos_y, self.width, self.width))
        py.display.update()

class input_field(ui_object):
    text = "1"
    font = py.font.Font(py.font.get_default_font(), 36)
    text_surface = py.surface

    def __init__(self, pos_x, pos_y, width, height, root):
        ui_object.__init__(self, pos_x, pos_y, width, height, root)
        self.update_text_object()
    
    def clicked(self, mouse_pos):
        if ui_object.clicked(self, mouse_pos):
            self.update_text_data()

    def update_text_data(self):
        while True:
            for event in py.event.get():
                if event.type == py.KEYDOWN:
                    try:
                        int(py.key.name(event.key))
                        self.text += py.key.name(event.key)
                    except:
                        pass
                    if event.key == py.K_RETURN or event.type == py.K_KP_ENTER:
                        return
                    elif event.key == py.K_BACKSPACE:
                        self.text = self.text[:-2]
                    self.update_text_object()
                elif event.type == py.MOUSEBUTTONDOWN:
                    if not ui_object.clicked(self, py.mouse.get_pos()):
                        return
                elif event.type == py.QUIT:
                    raise SystemExit()
                    
    def update_text_object(self):
        self.font = py.font.Font(py.font.get_default_font(), 36)
        self.text_surface = self.font.render(self.text, True, (0,0,0))
        self.rect = py.draw.rect(self.root, self.WHITE, (self.pos_x, self.pos_y, self.width, self.height))
        self.root.blit(self.text_surface, (self.pos_x, self.pos_y))
        py.display.update()