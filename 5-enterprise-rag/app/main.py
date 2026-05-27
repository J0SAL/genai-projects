from flask import Flask, request, render_template, jsonify
from models.vector_store import VectorStore
from services.enterprise_loader import EnterpriseDataLoader
from services.storage_service import S3Storage
from services.llm_service import LLMService
from services.rbac_service import RBACService
from config import Config
import os
import tempfile
import logging
from werkzeug.utils import secure_filename

# add code for donwload .env from gdrive - refer chatgpt


app = Flask(__name__)
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
rbac_service = RBACService(Config.ACCESS_POLICY_PATH)
enterprise_loader = EnterpriseDataLoader(rbac_service)
llm_service = LLMService(vector_store, rbac_service)

@app.route('/')
def index():
    return render_template('index.html')


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_document(file):
    """Process document based on file type and return text chunks"""
    temp_dir = tempfile.mkdtemp()
    safe_filename = secure_filename(file.filename)
    temp_path = os.path.join(temp_dir, safe_filename)
    
    try:
        file.save(temp_path)
        return enterprise_loader.load_file(temp_path, file.filename)
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        logger.debug("Upload endpoint called")
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        if not file.filename.endswith(('.txt', '.pdf', '.csv', '.json', '.sql')):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({'error': 'Only .txt, .pdf, .csv, .json, and .sql files are supported'}), 400

        logger.debug(f"Processing file: {file.filename}")
        user_id = request.form.get('user_id', 'alex_analyst')
        role = rbac_service.get_user_role(user_id)
        
        # Process the document
        try:
            text_chunks = process_document(file)
            logger.debug(f"Document processed into {len(text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500

        # Upload to S3
        s3_uploaded = False
        try:
            file.seek(0)  # Reset file pointer
            s3_uploaded = storage_service.upload_file(file, file.filename)
            logger.debug("File uploaded to S3" if s3_uploaded else "S3 upload skipped")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return jsonify({'error': f'Error uploading to S3: {str(e)}'}), 500

        # Add to vector store
        try:
            vector_store.add_documents(text_chunks)
            logger.debug("Documents added to vector store")
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            return jsonify({'error': f'Error adding to vector store: {str(e)}'}), 500

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'chunks_processed': len(text_chunks),
            'role': role,
            's3_uploaded': s3_uploaded
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    try:
        user_id = data.get('user_id', 'alex_analyst')
        role = rbac_service.get_user_role(user_id)
        response = llm_service.get_response(data['question'], user_id=user_id, role=role)
        return jsonify({
            'response': response['answer'],
            'citations': response['citations'],
            'confidence': response['confidence'],
            'trace': response['trace'],
            'role': role
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/users', methods=['GET'])
def users():
    return jsonify(rbac_service.get_users())


@app.route('/ingest-demo-data', methods=['POST'])
def ingest_demo_data():
    try:
        documents = enterprise_loader.load_directory(Config.ENTERPRISE_DATA_PATH)
        vector_store.add_documents(documents)
        return jsonify({
            'message': 'Demo enterprise data ingested successfully',
            'chunks_processed': len(documents)
        })
    except Exception as e:
        logger.error(f"Error ingesting demo data: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', '8080'))
    app.run(host='0.0.0.0', port=port, debug=True)
