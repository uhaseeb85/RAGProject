import os
import sys
from typing import List
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import requests

class PDFProcessor:
    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if os.path.exists(self.persist_directory):
            return FAISS.load_local(self.persist_directory, self.embeddings)
        return FAISS.from_texts(["Initialize vectorstore"], self.embeddings)

    def process_pdf(self, pdf_path: str):
        try:
            text = self.extract_text_from_pdf(pdf_path)
            self.add_text_to_vectorstore(text)
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    def add_text_to_vectorstore(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        self.vectorstore.add_texts(chunks)
        self.vectorstore.save_local(self.persist_directory)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        with open(pdf_path, "rb") as file:
            pdf = PdfReader(file)
            return "".join(page.extract_text() for page in pdf.pages)

class LMStudioLLM(OpenAI):
    def __init__(self, api_base):
        super().__init__(
            model_name="local_model",
            openai_api_base=api_base,
            openai_api_key="not_needed"
        )

    def completion_with_retry(self, **kwargs):
        try:
            data = {
                "messages": [{"role": "user", "content": kwargs["prompt"]}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 500)
            }
            response = requests.post(f"{self.openai_api_base}/v1/chat/completions", json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with LM Studio: {str(e)}")

class RAGApplication:
    def __init__(self, lm_studio_api_base: str):
        self.pdf_processor = PDFProcessor()
        self.llm = LMStudioLLM(api_base=lm_studio_api_base)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.pdf_processor.vectorstore.as_retriever(),
            memory=self.memory
        )

    def process_pdfs(self, pdf_paths: List[str], progress_callback=None):
        for i, pdf_path in enumerate(pdf_paths):
            try:
                self.pdf_processor.process_pdf(pdf_path)
                if progress_callback:
                    progress_callback(i + 1, len(pdf_paths))
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")

    def ask_question(self, question: str) -> str:
        try:
            response = self.qa_chain({"question": question})
            return response['answer']
        except Exception as e:
            return f"Error: {str(e)}"

class PDFProcessingThread(QThread):
    progress_updated = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, rag_app, pdf_paths):
        super().__init__()
        self.rag_app = rag_app
        self.pdf_paths = pdf_paths

    def run(self):
        self.rag_app.process_pdfs(self.pdf_paths, self.update_progress)
        self.finished.emit()

    def update_progress(self, current, total):
        self.progress_updated.emit(current, total)

class RAGApplicationGUI(QMainWindow):
    def __init__(self, lm_studio_api_base: str):
        super().__init__()
        self.rag_app = RAGApplication(lm_studio_api_base)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('RAG Application')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # PDF loading section
        pdf_layout = QHBoxLayout()
        self.pdf_label = QLabel('No PDFs loaded')
        pdf_layout.addWidget(self.pdf_label)
        
        load_button = QPushButton('Load PDFs')
        load_button.clicked.connect(self.load_pdfs)
        pdf_layout.addWidget(load_button)

        layout.addLayout(pdf_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Question input
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("Enter your question here...")
        layout.addWidget(self.question_input)

        # Answer button
        answer_button = QPushButton('Get Answer')
        answer_button.clicked.connect(self.get_answer)
        layout.addWidget(answer_button)

        # Answer display
        self.answer_display = QTextEdit()
        self.answer_display.setReadOnly(True)
        layout.addWidget(self.answer_display)

        central_widget.setLayout(layout)

    def load_pdfs(self):
        file_dialog = QFileDialog()
        pdf_paths, _ = file_dialog.getOpenFileNames(self, "Select PDF files", "", "PDF Files (*.pdf)")
        
        if pdf_paths:
            self.pdf_label.setText(f"Processing {len(pdf_paths)} PDF(s)")
            self.progress_bar.setVisible(True)
            self.processing_thread = PDFProcessingThread(self.rag_app, pdf_paths)
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.start()

    def update_progress(self, current, total):
        self.progress_bar.setValue(int(current / total * 100))

    def on_processing_finished(self):
        self.pdf_label.setText("PDFs processed and ready for questions")
        self.progress_bar.setVisible(False)

    def get_answer(self):
        question = self.question_input.toPlainText()
        if question:
            answer = self.rag_app.ask_question(question)
            self.answer_display.setText(answer)
        else:
            self.answer_display.setText("Please enter a question.")

def main():
    lm_studio_api_base = "http://localhost:1234"  # Update this with your LM Studio API base URL
    app = QApplication(sys.argv)
    ex = RAGApplicationGUI(lm_studio_api_base)
    ex.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()