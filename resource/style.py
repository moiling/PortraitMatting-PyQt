import os


def read_qss(style):
    style = '{}/resource/qss/{}.qss'.format(os.path.abspath(os.curdir), style)
    with open(style, 'r') as f:
        return f.read()


def icon_str(icon):
    return '{}/resource/icon/{}.png'.format(os.path.abspath(os.curdir), icon)


scroll_bar = read_qss('scroll_bar')
