import os


def read_qss(style):
    style = '{}/resource/qss/{}.qss'.format(os.path.abspath(os.curdir), style)
    with open(style, 'r') as f:
        return f.read()


def icon_str(icon):
    return '{}/resource/icon/{}.png'.format(os.path.abspath(os.curdir), icon)


scroll_bar = read_qss('scroll_bar')
tool_button_normal = read_qss('tool_button_normal')
tool_button_current = read_qss('tool_button_current')
pen_small_normal = read_qss('pen_small_normal')
pen_small_current = read_qss('pen_small_current')
pen_middle_normal = read_qss('pen_middle_normal')
pen_middle_current = read_qss('pen_middle_current')
pen_large_normal = read_qss('pen_large_normal')
pen_large_current = read_qss('pen_large_current')
