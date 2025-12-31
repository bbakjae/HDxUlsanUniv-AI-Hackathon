"""
Simple Gradio test
"""
import gradio as gr

def simple_search(query):
    return f"You searched for: {query}"

# Create simple UI
with gr.Blocks(title="Test") as demo:
    gr.Markdown("# Test Search")
    query_input = gr.Textbox(label="Search")
    output = gr.Textbox(label="Result")
    btn = gr.Button("Search")
    btn.click(fn=simple_search, inputs=query_input, outputs=output)

print("Launching Gradio...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
