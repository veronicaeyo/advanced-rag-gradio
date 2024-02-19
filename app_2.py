import gradio as gr

css = """ 
h1 {
    text-align: center;
    display:block;
    }
    """
with gr.Blocks(css=css) as demo:
    gr.Markdown("# ADVANCED RAG GPT")

    with gr.Tab("Basic RAG"):
        gr.Markdown(
            "Basic RAG is a simplified version of the RAG (Retrieval-Augmented Generation) model. It uses a retrieval model to find relevant documents and then generates a response based on the retrieved information."
        ),
        gr.Dropdown(["Context Relevance: High", "Answer Relevance: Medium", "Answer Coherence: Medium"], label="TruLens Evaluation"),

        chatbot = gr.Chatbot(show_copy_button=True)
        text = gr.Textbox()

        with gr.Column():
            file = gr.File(
                type="filepath",
                label="Upload file",
                height=80,
                file_types=["text", ".pdf"],
            )

        file.upload(fn=lambda x: x.name, inputs=file, outputs=text)

    with gr.Tab("Sentence Window"):

        gr.Markdown(
            "Sentence Window is a technique used in RAG models to limit the context window for retrieval and generation. It allows the model to focus on a specific range of sentences in the document."
        ),
        gr.Markdown(
            "TruLens Evaluation: - Context Relevance: High - Answer Relevance: High - Answer Coherence: Medium"
        )

        chatbot = gr.Chatbot(show_copy_button=True)
        text = gr.Textbox()

        with gr.Column():
            file = gr.File(
                type="filepath",
                label="Upload file",
                height=80,
                file_types=["text", ".pdf"],
            )

        file.upload(fn=lambda x: x.name, inputs=file, outputs=text)

    with gr.Tab("Auto-Merging"):
        gr.Markdown(
            "Auto-Merging is a feature in RAG models that automatically merges similar answers to improve the quality and coherence of the generated responses."
        ),
        gr.Markdown(
            "TruLens Evaluation: - Context Relevance: medium - Answer Relevance: high - Answer Coherence: High"
        )

        chatbot = gr.Chatbot(show_copy_button=True)
        text = gr.Textbox()

        with gr.Column():
            file = gr.File(
                type="filepath",
                label="Upload file",
                height=80,
                file_types=["text", ".pdf"],
            )

        file.upload(fn=lambda x: x.name, inputs=file, outputs=text)

if __name__ == "__main__":
    demo.queue().launch()