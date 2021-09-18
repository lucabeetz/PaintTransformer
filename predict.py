import sys
import os
import tempfile
from pathlib import Path
import shutil
import glob
from PIL import Image
import cog

sys.path.insert(0, "inference")

from inference import run_inference


class Predictor(cog.Predictor):
    def setup(self):
        self.basepath = os.getcwd()

    @cog.input("image", type=Path, help="input image")
    @cog.input(
        "output_type",
        type=str,
        default="gif",
        options=["png", "gif"],
        help="choose output the final png or a gif with the painting process",
    )
    def predict(self, image, output_type="gif"):
        basename = os.path.basename(str(image))

        # avoid subdirectory import issue
        os.chdir("./inference")
        output_dir = "output"

        run_inference(
            input_path=str(image),
            model_path="model.pth",
            output_dir=output_dir,  # whether need intermediate results for animation.
            need_animation=True if output_type == "gif" else False,
            resize_h=None,  # resize original input to this size. None means do not resize.
            resize_w=None,
            serial=True,
        )

        os.chdir(self.basepath)

        if output_type == "gif":
            # Set to dir with output images
            in_dir = os.path.join(
                "inference", output_dir, os.path.splitext(basename)[0] + "/*.jpg"
            )
            out_path = Path(tempfile.mkdtemp()) / "out.gif"

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(in_dir))]
            img.save(
                fp=str(out_path),
                format="GIF",
                append_images=imgs,
                save_all=True,
                duration=100,
                loop=0,
            )
        else:

            img = Image.open(os.path.join("inference", output_dir, basename))
            out_path = Path(tempfile.mkdtemp()) / "out.gif"
            img.save(str(out_path))

        clean_folder(os.path.join("inference", output_dir))
        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
