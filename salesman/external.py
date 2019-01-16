import imageio
import glob
import os


def images_to_animated_gif(glob_pattern,
                           gif_filename='./movie.gif',
                           frame_length=0.1,
                           end_pause=1,
                           step=1):
    # based on https://stackoverflow.com/a/35943809/3247880

    # frame_length  - seconds between frames
    # end_pause     - seconds to stay on last frame
    # step = 10     - include only every n=step file

    images = []
    image = None
    filenames = glob.glob(glob_pattern)

    # append frames of movie
    n_files = filenames
    for filename in filenames[0::step]:
        image = imageio.imread(filename)
        images.append(image)
        print('.', end='')

    if image is not None:
        # repeat last frame
        for rep in range(int(end_pause / frame_length)):
            images.append(image)

    # create animated gif
    print(f'\nsaving animation to: {gif_filename}')
    imageio.mimsave(gif_filename, images, 'GIF', duration=frame_length)
    s = os.path.getsize(gif_filename)
    print(f"image saved. Size: {sizeof_fmt(s)}, " +
          f"Duration: {frame_length * len(images)}" +
          f" sec., " +
          f"#Frames: {len(images)}")


def sizeof_fmt(num, suffix='B'):
# https://stackoverflow.com/a/1094933/3247880
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


if __name__ == "__main__":
    images_to_animated_gif(glob_pattern='./res/circle-len/*.png',
                           gif_filename='./c-len_animation.gif',
                           frame_length=0.2,
                           end_pause=1,
                           step=10,
                           )

    images_to_animated_gif(glob_pattern='./res/circle-route/*.png',
                           gif_filename='./c-tour_animation.gif',
                           frame_length=0.1,
                           end_pause=2,
                           step=5,
                           )

    images_to_animated_gif(glob_pattern='./res/rand-len/*.png',
                           gif_filename='./r-len_animation.gif',
                           frame_length=0.2,
                           end_pause=1,
                           step=10,
                           )

    images_to_animated_gif(glob_pattern='./res/rand-route/*.png',
                           gif_filename='./r-tour_animation.gif',
                           frame_length=0.1,
                           end_pause=2,
                           step=5,
                           )
