import gradio as gr

css = """
h1 {
    text-align: center;
    display:block;
}
"""
#! Take care of multiple files

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "# ADVANCED RAG GPT \n"
        "This Gradio app is powered by LlamaIndex's `ReActAgent` with\n"
        "OpenAI's GPT-4-Turbo as the LLM. The tools are listed below.\n"
        "## Tools\n"
    )

    # basic, sentence window, auto merging
    # trulens eval - context relevance, groundedness, answer relevance

    chatbot = gr.Chatbot(show_copy_button=True)

    with gr.Column():
        text = gr.Textbox()

        file = gr.File(
            type="filepath",
            label="Upload a file",
            height=80,
            file_types=["text", ".pdf"],
        )

    file.upload(fn=lambda x: x.name, inputs=file, outputs=text)

if __name__ == "__main__":
    demo.queue().launch()
