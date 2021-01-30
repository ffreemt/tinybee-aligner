"""Testing gradio."""
# pylint: disable=invalid-name

from pathlib import Path
import gradio as gr


def disp_text(file):
    """Test."""
    # print(file)
    file_cont = Path(file.name).read_text("utf8")
    # return f"{file.name}: " + str(type(file))
    # return file_cont.splitlines()
    return file_cont


# image = gr.inputs.Image(shape=(299, 299, 3))
file = gr.inputs.File(label="file1")

label = gr.outputs.Label(num_top_classes=3)

# launching the interface
iface = gr.Interface(
    # fn=classify_image,
    fn=disp_text,
    # inputs="file",
    # inputs="textbox",
    inputs=file,
    outputs=gr.outputs.HighlightedText(
        color_map={"+": "lightgreen", "-": "pink", " ": "none",}
    ),
    capture_session=True,
    interpretation="default",
)  # .launch()

iface.test_launch()

if __name__ == "__main__":
    iface.launch(share=0)
