# 🔄 Corrective RAG Agent

A sophisticated Retrieval-Augmented Generation (RAG) system that implements a corrective multi-stage workflow using LangGraph. This advanced system combines document retrieval, relevance grading, query transformation, and web search to provide comprehensive and accurate responses through intelligent self-correction mechanisms.

## 🌟 Features

### Core Functionality
- **Smart Document Retrieval**: Uses Qdrant vector store for efficient document retrieval
- **Document Relevance Grading**: Employs Claude 3.5 sonnet to assess document relevance
- **Query Transformation**: Improves search results by optimizing queries when needed
- **Web Search Fallback**: Uses Tavily API for web search when local documents aren't sufficient
- **Multi-Model Approach**: Combines OpenAI embeddings and Claude 3.5 sonnet for different tasks
- **Interactive UI**: Built with Streamlit for easy document upload and querying

### Advanced Capabilities
- **Self-Correction**: Automatically identifies and corrects retrieval issues
- **Relevance Assessment**: Intelligent scoring of document relevance
- **Query Optimization**: Dynamic query refinement for better results
- **Multi-Source Integration**: Seamlessly combines local and web sources
- **Workflow Visualization**: Step-by-step process visualization
- **Quality Assurance**: Built-in quality checks and validation

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Anthropic API key
- Tavily API key
- Qdrant Cloud account

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rchhabra13/portfolio-projects.git
   cd portfolio-projects/corrective_rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   export TAVILY_API_KEY="your-tavily-api-key-here"
   ```

4. **Set up Qdrant Cloud**
   - Visit [Qdrant Cloud](https://cloud.qdrant.io/)
   - Create an account or sign in
   - Create a new cluster
   - Get your credentials:
     - Qdrant API Key: Found in API Keys section
     - Qdrant URL: Your cluster URL (format: `https://xxx-xxx.aws.cloud.qdrant.io`)

5. **Run the application**
   ```bash
   streamlit run corrective_rag.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Upload documents or provide URLs
   - Start asking questions and see the corrective RAG process

## 💡 Usage Examples

### Document Analysis
```
"What are the main findings in this research paper?"
"Summarize the key points from the business report."
"What methodology was used in this study?"
```

### Complex Queries
```
"Compare the different approaches mentioned in these documents."
"What are the limitations of the current methods discussed?"
"How do these findings relate to recent developments in the field?"
```

### Research Queries
```
"Find information about machine learning algorithms in these documents."
"What are the technical specifications mentioned?"
"What are the main recommendations for future work?"
```

## 🛠️ Technical Architecture

### Core Technologies
- **Framework**: LangChain for RAG orchestration
- **Workflow Management**: LangGraph for multi-stage processing
- **Vector Database**: Qdrant for document storage and retrieval
- **Language Models**: Claude 3.5 sonnet for analysis and generation
- **Embeddings**: OpenAI text-embedding-3-small
- **Web Search**: Tavily API for web search capabilities
- **UI**: Streamlit for user interface

### Corrective RAG Pipeline
1. **Document Upload**: Documents are uploaded and processed
2. **Text Extraction**: Text is extracted and chunked
3. **Embedding Generation**: Text chunks are converted to embeddings
4. **Vector Storage**: Embeddings are stored in Qdrant
5. **Query Processing**: User queries are processed and analyzed
6. **Initial Retrieval**: Relevant documents are retrieved
7. **Relevance Grading**: Claude 3.5 assesses document relevance
8. **Query Transformation**: Queries are optimized if needed
9. **Web Search Fallback**: Web search is performed if local docs insufficient
10. **Response Generation**: Final answer is generated and validated

## 📊 Supported Document Types

### Academic Papers
- Research papers and studies
- Conference proceedings
- Journal articles
- Theses and dissertations

### Business Documents
- Reports and whitepapers
- Financial statements
- Legal documents
- Policy documents

### Technical Documentation
- User manuals
- API documentation
- Technical specifications
- Code documentation

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
TAVILY_API_KEY=your-tavily-api-key
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_URL=your-qdrant-url
```

### Customization Options
- **Chunk Size**: Adjust text chunk size for better retrieval
- **Chunk Overlap**: Set overlap between chunks
- **Top K Results**: Number of relevant chunks to retrieve
- **Relevance Threshold**: Minimum relevance score for documents
- **Web Search Depth**: Control web search comprehensiveness
- **Response Length**: Limit response detail level

## 📈 Performance Features

- **Efficient Vector Search**: Optimized similarity search using Qdrant
- **Intelligent Caching**: Reduces redundant processing
- **Parallel Processing**: Concurrent document and web processing
- **Memory Management**: Efficient handling of large document collections
- **Error Recovery**: Robust error handling and fallback mechanisms
- **Quality Validation**: Built-in quality checks and validation

## 🔒 Security & Privacy

- **Data Encryption**: All data encrypted in transit and at rest
- **API Security**: Secure API key management
- **Database Security**: Secure database connections
- **Privacy Compliance**: Follows data privacy best practices

## 🤝 Contributing

We welcome contributions from developers and AI enthusiasts:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- Additional document format support
- Improved retrieval algorithms
- Better web search integration
- Enhanced UI/UX
- Performance optimizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the FAQ section

## 🔮 Roadmap

- [ ] Support for more document formats
- [ ] Advanced web search capabilities
- [ ] Multi-language support
- [ ] Real-time document updates
- [ ] Collaborative features
- [ ] API endpoint for integration
- [ ] Mobile app support
- [ ] Advanced analytics

## 📊 Use Cases

### Research & Academia
- Literature review assistance
- Research paper analysis
- Academic document summarization
- Knowledge synthesis

### Business & Legal
- Document analysis and review
- Market research integration
- Legal document processing
- Business intelligence

### Education
- Interactive learning materials
- Document-based Q&A
- Research assistance
- Knowledge exploration

### Personal Use
- Document organization
- Information retrieval
- Content summarization
- Research assistance

## 🙏 Acknowledgments

- Anthropic for the Claude 3.5 sonnet model
- OpenAI for the embedding models
- LangChain and LangGraph for the RAG framework
- Qdrant for vector storage
- Tavily for web search capabilities
- Streamlit for the user interface

---

**Note**: This application is designed for educational and research purposes. Always ensure you have the right to process and analyze the documents you upload.

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->
