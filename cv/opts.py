import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='Doomscrolling Detection')

    ## Paths
    parser.add_argument('--record-video', action='store_true',
                        help='record video')
    parser.add_argument('--headless', action='store_true',
                        help='run in headless mode, no display')
    parser.add_argument('--reclined-threshold', type=float, default=0.2,
                        help='threshold for reclined detection')
    parser.add_argument('--holding-phone-threshold', type=float, default=0.2,
                        help='threshold for holding phone detection')

    opts = parser.parse_args()
    return opts