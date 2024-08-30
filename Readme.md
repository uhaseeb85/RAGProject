# RAG Application

The RAG (Retrieval-Augmented Generation) Application is a Python-based tool that allows users to load PDF files, process them, and ask questions related to the content of the PDFs. The application uses natural language processing techniques and a language model to generate answers based on the information retrieved from the processed PDFs.

## Features

- Load and process multiple PDF files
- Extract text from PDFs and store it in a vector store
- Ask questions related to the content of the loaded PDFs
- Generate answers using a language model and the retrieved information
- Graphical user interface (GUI) for easy interaction

## Requirements

- Python 3.7 or higher
- PyQt6
- PyPDF2
- Langchain
- Hugging Face Embeddings
- FAISS
- OpenAI API key (for the language model)
- LM Studio API (for the custom language model)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/rag-application.git
   ```

2. Navigate to the project directory:
   ```
   cd rag-application
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # For Unix/Linux
   venv\Scripts\activate  # For Windows
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up the OpenAI API key:
   - Sign up for an OpenAI account and obtain an API key.
   - Set the `OPENAI_API_KEY` environment variable with your API key.

6. Set up the LM Studio API:
   - Make sure you have access to an LM Studio API endpoint.
   - Update the `lm_studio_api_base` variable in the `main` function of `RagApp.py` with your LM Studio API base URL.

## Usage

1. Run the application:
   ```
   python RagApp.py
   ```

2. The RAG Application GUI will open.

3. Click on the "Load PDFs" button to select PDF files you want to process. You can select multiple files at once.

4. The application will start processing the selected PDFs. The progress will be displayed in the progress bar.

5. Once the PDFs are processed, you can enter a question related to the content of the PDFs in the question input field.

6. Click on the "Get Answer" button to generate an answer based on the information retrieved from the processed PDFs.

7. The generated answer will be displayed in the answer display area.

8. You can continue asking questions related to the loaded PDFs.

## Customization

- If you want to use a different vector store or embeddings, you can modify the `PDFProcessor` class in `RagApp.py` accordingly.

- If you want to customize the language model or use a different one, you can modify the `LMStudioLLM` class in `RagApp.py` or replace it with another language model implementation.

- The GUI can be customized by modifying the `RAGApplicationGUI` class in `RagApp.py` to change the layout, styles, or add additional features.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Langchain](https://github.com/hwchase17/langchain) for the retrieval and question-answering capabilities.
- [Hugging Face](https://huggingface.co/) for the embeddings model.
- [OpenAI](https://openai.com/) for the language model.
- [LM Studio](https://lm-studio.com/) for the custom language model API.