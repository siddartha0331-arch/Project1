import streamlit as st
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import warnings
import os
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== PAGE CONFIGURATION ==================

st.set_page_config(
    page_title="🎭 Emotion Detection AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM STYLING ==================

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header with gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.95;
    }
    
    /* Colored sentence display */
    .colored-sentence {
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 500;
        line-height: 1.6;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Emotion result box */
    .emotion-result-box {
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .emotion-result-box:hover {
        transform: translateY(-5px);
    }
    
    .emotion-anger {
        background: linear-gradient(135deg, #FF6B6B 0%, #C92A2A 100%);
        color: white;
    }
    
    .emotion-disgust {
        background: linear-gradient(135deg, #9B59B6 0%, #6C3483 100%);
        color: white;
    }
    
    .emotion-fear {
        background: linear-gradient(135deg, #FFA502 0%, #E67E22 100%);
        color: white;
    }
    
    .emotion-joy {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
        color: white;
    }
    
    .emotion-neutral {
        background: linear-gradient(135deg, #95A5A6 0%, #7F8C8D 100%);
        color: white;
    }
    
    .emotion-sadness {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white;
    }
    
    .emotion-surprise {
        background: linear-gradient(135deg, #F1C40F 0%, #F39C12 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1.1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ================== CACHE FUNCTIONS FOR PERFORMANCE ==================

@st.cache_resource
def load_bert_model():
    """Load BERT-BiLSTM model (cached)"""
    import os
    import requests
    
    # Check if model exists locally
    model_path = 'best_optimized_bert_bilstm_model.h5'
    
    if not os.path.exists(model_path):
        st.info("📥 Model file not found locally. Downloading from Dropbox...")
        
        # Try multiple Dropbox URL formats
        DROPBOX_URLS = [
            # Format 1: dropboxusercontent (usually most reliable)
            "https://dl.dropboxusercontent.com/scl/fi/5h6slzbxcqox0mun7i3e6/best_optimized_bert_bilstm_model.h5?rlkey=a1tuqyreja1d96uhe1ta5ua0o&dl=1",
            # Format 2: Original link
            "https://www.dropbox.com/scl/fi/5h6slzbxcqox0mun7i3e6/best_optimized_bert_bilstm_model.h5?rlkey=a1tuqyreja1d96uhe1ta5ua0o&st=cp3k1o4v&dl=1",
            # Format 3: Without st parameter
            "https://www.dropbox.com/scl/fi/5h6slzbxcqox0mun7i3e6/best_optimized_bert_bilstm_model.h5?rlkey=a1tuqyreja1d96uhe1ta5ua0o&dl=1",
        ]
        
        download_success = False
        
        for idx, url in enumerate(DROPBOX_URLS):
            try:
                st.info(f"🔄 Trying download method {idx + 1}/{len(DROPBOX_URLS)}...")
                
                with st.spinner(f"Downloading model (33.8 MB)..."):
                    # Add headers to mimic browser request
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(url, stream=True, headers=headers, timeout=300)
                    response.raise_for_status()
                    
                    # Check if we got HTML instead of file (Dropbox error page)
                    content_type = response.headers.get('content-type', '')
                    if 'text/html' in content_type:
                        st.warning(f"Method {idx + 1} returned HTML page instead of file. Trying next method...")
                        continue
                    
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # If no content-length or too small, skip
                    if total_size > 0 and total_size < 10_000_000:  # Less than 10MB
                        st.warning(f"Method {idx + 1} file too small ({total_size / (1024*1024):.1f} MB). Expected ~33.8 MB.")
                        continue
                    
                    block_size = 1024 * 1024  # 1MB chunks
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(model_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = min(downloaded / total_size, 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Verify download
                    if os.path.exists(model_path):
                        file_size = os.path.getsize(model_path) / (1024 * 1024)
                        if file_size > 30:  # Should be around 33.8 MB
                            st.success(f"✅ Model downloaded successfully! ({file_size:.1f} MB)")
                            download_success = True
                            break
                        else:
                            st.warning(f"Method {idx + 1} downloaded file too small ({file_size:.1f} MB). Trying next method...")
                            os.remove(model_path)
                            continue
                            
            except requests.exceptions.RequestException as e:
                st.warning(f"Method {idx + 1} failed: {str(e)}")
                continue
            except Exception as e:
                st.warning(f"Method {idx + 1} unexpected error: {str(e)}")
                continue
        
        if not download_success:
            st.error("❌ All download methods failed!")
            st.warning("""
            💡 **Manual Upload Required:**
            
            Upload the model file directly to your GitHub repository using Git command line:
            
            ```bash
            cd your-project-folder
            git add best_optimized_bert_bilstm_model.h5
            git commit -m "Add model file"
            git push
            ```
            """)
            return None
    else:
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        st.info(f"✅ Model file found locally ({file_size:.1f} MB)")
    
    try:
        # Rebuild model architecture
        from transformers import TFAutoModel
        from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, BatchNormalization, Lambda
        from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
        from tensorflow.keras.models import Model
        
        with st.spinner("🔄 Loading BERT base model..."):
            bert_model = TFAutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, from_pt=True)
            bert_config = bert_model.config
        
        max_len = 128
        num_emotion_classes = 7
        num_sentiment_classes = 3
        
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
        token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
        
        def call_bert(inputs):
            ids, mask, token_ids = inputs
            return bert_model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids)[0]
        
        sequence_output = Lambda(
            call_bert,
            output_shape=(max_len, bert_config.hidden_size),
            name='bert_lambda'
        )([input_ids, attention_mask, token_type_ids])
        
        bilstm = Bidirectional(
            LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3),
            name='bilstm_1'
        )(sequence_output)
        bilstm = Dropout(0.3, name='bilstm_dropout_1')(bilstm)
        bilstm = BatchNormalization()(bilstm)
        
        bilstm = Bidirectional(
            LSTM(128, return_sequences=True, recurrent_dropout=0.3, dropout=0.3),
            name='bilstm_2'
        )(bilstm)
        bilstm = Dropout(0.3, name='bilstm_dropout_2')(bilstm)
        bilstm = BatchNormalization()(bilstm)
        
        bilstm_avg = GlobalAveragePooling1D()(bilstm)
        bilstm_max = GlobalMaxPooling1D()(bilstm)
        bilstm_final = Concatenate()([bilstm_avg, bilstm_max])
        
        shared = Dense(256, activation='relu', kernel_initializer='he_normal', name='shared_dense_1')(bilstm_final)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3, name='shared_dropout_1')(shared)
        
        shared = Dense(128, activation='relu', kernel_initializer='he_normal', name='shared_dense_2')(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3, name='shared_dropout_2')(shared)
        
        shared = Dense(64, activation='relu', kernel_initializer='he_normal', name='shared_dense_3')(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3, name='shared_dropout_3')(shared)
        
        emotion_dense = Dense(64, activation='relu', kernel_initializer='he_normal', name='emotion_dense_1')(shared)
        emotion_dense = BatchNormalization()(emotion_dense)
        emotion_dense = Dropout(0.3, name='emotion_dropout_1')(emotion_dense)
        
        emotion_dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='emotion_dense_2')(emotion_dense)
        emotion_dense = Dropout(0.2, name='emotion_dropout_2')(emotion_dense)
        
        emotion_output = Dense(num_emotion_classes, activation='softmax', name='emotion_output')(emotion_dense)
        
        sentiment_dense = Dense(64, activation='relu', kernel_initializer='he_normal', name='sentiment_dense_1')(shared)
        sentiment_dense = BatchNormalization()(sentiment_dense)
        sentiment_dense = Dropout(0.3, name='sentiment_dropout_1')(sentiment_dense)
        
        sentiment_dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='sentiment_dense_2')(sentiment_dense)
        sentiment_dense = Dropout(0.2, name='sentiment_dropout_2')(sentiment_dense)
        
        sentiment_output = Dense(num_sentiment_classes, activation='softmax', name='sentiment_output')(sentiment_dense)
        
        model = Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=[emotion_output, sentiment_output],
            name='BERT_BiLSTM_MultiTask'
        )
        
        with st.spinner("🔄 Loading model weights..."):
            try:
                # Try loading weights directly using h5py (avoids Keras compatibility issues)
                import h5py
                
                with h5py.File(model_path, 'r') as f:
                    if 'model_weights' in f.keys():
                        # Newer format
                        weights_group = f['model_weights']
                    else:
                        # Older format - weights at root
                        weights_group = f
                    
                    # Load weights layer by layer
                    weights_loaded = 0
                    for layer in model.layers:
                        layer_name = layer.name
                        if layer_name in weights_group.keys():
                            layer_weights_group = weights_group[layer_name]
                            
                            # Get weight names for this layer
                            weight_names = [n.decode('utf8') if hasattr(n, 'decode') else n 
                                          for n in layer_weights_group.attrs.get('weight_names', [])]
                            
                            if weight_names:
                                # Load weights
                                weight_values = [layer_weights_group[weight_name][()] 
                                               for weight_name in weight_names]
                                layer.set_weights(weight_values)
                                weights_loaded += 1
                    
                    st.success(f"✅ Loaded weights for {weights_loaded} layers using h5py")
                
            except Exception as h5_error:
                st.warning(f"h5py method failed: {h5_error}. Trying alternative method...")
                
                # Fallback: Try loading with custom deserialization
                try:
                    # Build custom config to handle legacy parameters
                    import tensorflow.keras.backend as K
                    
                    # Temporarily monkey-patch Input layer to ignore legacy params
                    original_input = tf.keras.layers.Input
                    
                    def patched_input(*args, **kwargs):
                        # Remove problematic legacy parameters
                        kwargs.pop('batch_shape', None)
                        kwargs.pop('optional', None)
                        return original_input(*args, **kwargs)
                    
                    tf.keras.layers.Input = patched_input
                    
                    # Now try loading
                    custom_objects = {'Lambda': Lambda, 'call_bert': call_bert}
                    temp_model = load_model(model_path, custom_objects=custom_objects, compile=False)
                    
                    # Restore original Input
                    tf.keras.layers.Input = original_input
                    
                    # Copy weights
                    weights_loaded = 0
                    for layer in model.layers:
                        try:
                            temp_layer = temp_model.get_layer(layer.name)
                            if temp_layer is not None:
                                weights = temp_layer.get_weights()
                                if weights:
                                    layer.set_weights(weights)
                                    weights_loaded += 1
                        except (ValueError, AttributeError):
                            continue
                    
                    del temp_model
                    st.success(f"✅ Loaded weights for {weights_loaded} layers using fallback method")
                    
                except Exception as fallback_error:
                    st.error(f"❌ Both methods failed!")
                    st.error(f"h5py error: {h5_error}")
                    st.error(f"Fallback error: {fallback_error}")
                    raise fallback_error
        
        model.compile(
            loss=['categorical_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.7, 0.3],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
            metrics=[['accuracy'], ['accuracy']]
        )
        
        st.success(f"✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        return None

@st.cache_resource
def load_bert_tokenizer():
    """Load BERT tokenizer (cached)"""
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        st.success("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        return None

# ================== PREDICTION FUNCTION ==================

def predict_emotion_sentiment(text, model, tokenizer, max_len=128):
    """Predict emotion and sentiment for text"""
    
    if not text.strip():
        return None
    
    try:
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded.get('token_type_ids', np.zeros_like(input_ids))
        
        emotion_pred, sentiment_pred = model.predict(
            [input_ids, attention_mask, token_type_ids],
            verbose=0
        )
        
        emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        emotion_pred = emotion_pred[0]
        emotion_idx = np.argmax(emotion_pred)
        emotion = emotion_labels[emotion_idx]
        emotion_confidence = emotion_pred[emotion_idx]
        emotion_probs = {emotion_labels[i]: float(emotion_pred[i]) for i in range(len(emotion_labels))}
        
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment_pred = sentiment_pred[0]
        sentiment_idx = np.argmax(sentiment_pred)
        sentiment = sentiment_labels[sentiment_idx]
        sentiment_confidence = sentiment_pred[sentiment_idx]
        sentiment_probs = {sentiment_labels[i]: float(sentiment_pred[i]) for i in range(len(sentiment_labels))}
        
        emotion_emojis = {
            'anger': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'joy': '😊',
            'neutral': '😐',
            'sadness': '😢',
            'surprise': '😲'
        }
        
        sentiment_emojis = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }
        
        return {
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'emotion_probs': emotion_probs,
            'emotion_emoji': emotion_emojis[emotion],
            'sentiment': sentiment,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_probs': sentiment_probs,
            'sentiment_emoji': sentiment_emojis[sentiment]
        }
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def get_emotion_color(emotion):
    """Get color for emotion"""
    emotion_colors = {
        'anger': '#FF6B6B',
        'disgust': '#9B59B6',
        'fear': '#FFA502',
        'joy': '#2ECC71',
        'neutral': '#95A5A6',
        'sadness': '#3498DB',
        'surprise': '#F1C40F'
    }
    return emotion_colors.get(emotion, '#95A5A6')

# ================== MAIN APP ==================

def main():
    
    st.markdown("""
    <div class='main-header'>
        <h1>🎭 AI Emotion Detection System</h1>
        <p>Advanced deep learning model for analyzing emotions and sentiments in text using BERT-BiLSTM architecture</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🤖 About This Project")
    st.sidebar.info("""
    **BERT-BiLSTM Hybrid Model**
    
    - 7 Emotion Classes
    - 3 Sentiment Classes  
    - Context-aware analysis
    - High accuracy: 80-88%
    """)
    
    with st.spinner("🔄 Loading AI model and tokenizer..."):
        model = load_bert_model()
        tokenizer = load_bert_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("⚠️ Failed to load model or tokenizer. Please check your setup.")
        st.stop()
    
    st.markdown("### 📝 Enter Your Text for Analysis")
    
    text_input = st.text_area(
        "",
        height=150,
        placeholder="Type or paste your text here...\n\nExamples:\n• I'm so happy and excited about this!\n• This situation makes me really angry.\n• I'm feeling anxious about tomorrow.",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🚀 Analyze Emotion", use_container_width=True)
    
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()
    
    if analyze_button and text_input.strip():
        with st.spinner("🔍 Analyzing..."):
            result = predict_emotion_sentiment(text_input, model, tokenizer)
        
        if result:
            st.markdown("---")
            
            emotion_color = get_emotion_color(result['emotion'])
            st.markdown(f"""
            <div class='colored-sentence' style='background-color: {emotion_color}; color: white;'>
                "{text_input}"
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='emotion-result-box emotion-{result['emotion']}'>
                {result['emotion_emoji']} Detected Emotion: {result['emotion'].upper()} {result['emotion_emoji']}
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"Emotion: {result['emotion'].title()} ({result['emotion_confidence']:.1%} confidence)")
            st.info(f"Sentiment: {result['sentiment'].title()} ({result['sentiment_confidence']:.1%} confidence)")

if __name__ == "__main__":
    main()
