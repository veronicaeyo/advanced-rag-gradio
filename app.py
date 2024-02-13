import gradio as gr

css = """ 
h1 {
    text-align: center;
    display:block;
    }
    """
with gr.Blocks(css=css) as demo:
    gr.Markdown("# ADVANCED RAG GPT")

    # basic, sentence_window, auto_merging
    # TruLens Eval - context relevance, answer relevance, and answer relevance
    # make research  on how to organise the markdown on gradio a give information on basic etc, put an image of then too and also add the trulens eval of each one

    chatbot = gr.Chatbot(show_copy_button=True)
    text = gr.Textbox()

    with gr.Column():
        file = gr.File(
            type="filepath", label="Upload file", height=80, file_types=["text", ".pdf"]
        )

    file.upload(fn=lambda x: x.name, inputs=file, outputs=text)


if __name__ == "__main__":
    demo.queue().launch()
