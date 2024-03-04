import gradio as gr
from typing import List, Tuple

from backend import ChatbotInterface


def add_text(chat_history: List[Tuple[str, str]], query: str):
    chat_history += [(query, "")]
    return chat_history, gr.update(value="", interactive=False)


chatbot_interface = ChatbotInterface()


css = """
h1 {
    text-align: center;
    display:block;
}

#upload{height: 120px; overflow-y: scroll !important; }
"""
#! Take care of multiple files

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "# ADVANCED RAG GPT \n"
        "### Basic RAG: The document is loaded and the text is splitted into smaller chunks and embedded. The embeddings is stored in a vector store index. When a query is issued, a similarity search is done and the relevant chunk is fetched from the vector storage and sent to the LLM.  \n"
        "### Sentence Window Retrieval: Sentence is split into chunks and embedded, then stored in a vector store index with the context of the sentence that occured before and after it. At retrieval, the relevant sentence chunk is retrieved from the vectore storage, along with the full surrounding context and fed to the LLM. \n"
        "### Auto-merging Retrieval: Sentence is split into parent chunks and children chunks, whenever the child chunk is retrieved, the parent chunk is also  retrieved to provide more context for the LLM. \n"
    )
    with gr.Row():
        chatbot = gr.Chatbot(
            show_copy_button=True,
            scale=2,
            height="460px",
            avatar_images=("images/user.jpg", "images/bot.png"),
        )
        console = gr.TextArea(
            show_copy_button=True,
            lines=18,
            max_lines=18,
            label="Console",
            info="Contains token usage, similarity search result and other info from RAG",
        )

    with gr.Row():
        query = gr.Textbox(
            label="Question",
            placeholder="Input your question......",
            show_copy_button=True,
            scale=4,
        )

        file = gr.File(
            type="filepath",
            label="Upload a file",
            height=80,
            file_types=["text", ".pdf"],
            elem_id="upload",
        )

        rag_type = gr.Dropdown(
            chatbot_interface.rag_types,
            value="basic",
            label="Rag_Type",
            info="Choose the type of RAG model to use",
            max_choices=1,
        )
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.ClearButton([query, chatbot, console], variant="primary")

    with gr.Accordion(label="Additional parametrs for Sentence Window Rag"):
        with gr.Row():
            window_size = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                label="Window Size",
                value=3,
                info="Number of Sentences to retrieve around text chunks",
            )

            top_n = gr.Slider(
                minimum=2,
                maximum=4,
                step=1,
                label="Rerank Top N",
                value=2,
                info="Top N for reranking the results from similarity search",
            )

        with gr.Accordion(label="Additional parametrs for Auto merging Rag"):
            top_n = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                label="Rerank Top N",
                value=2,
                info="Top N for reranking the results from similarity search",
            )

    gr.on(
        triggers=[clear_btn.click, file.clear],
        fn=chatbot_interface.reset_chat,
        inputs=None,
        outputs=[chatbot, console, query],
    )

    response = gr.on(
        triggers=[submit_btn.click, query.submit],
        fn=add_text,
        inputs=[chatbot, query],
        outputs=[chatbot, query],
    ).then(
        fn=chatbot_interface.generate_response,
        inputs=[
            file,
            chatbot,
            rag_type,
            window_size,
            top_n,
        ],
        outputs=[chatbot, console],
    )

    response.then(lambda: gr.update(interactive=True), None, [query], queue=False)


if __name__ == "__main__":
    demo.queue().launch()

    # basic, sentence window, auto merging
    # trulens eval - context relevance, groundedness, answer relevance
