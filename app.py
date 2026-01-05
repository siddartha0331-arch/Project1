import streamlit as st
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
    page_title="🎭 Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM STYLING ==================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .emotion-anger {
        background-color: #FF0000;
        color: white;
    }
    .emotion-disgust {
        background-color: #8B008B;
        color: white;
    }
    .emotion-fear {
        background-color: #FFA500;
        color: white;
    }
    .emotion-joy {
        background-color: #00FF00;
        color: black;
    }
    .emotion-neutral {
        background-color: #808080;
        color: white;
    }
    .emotion-sadness {
        background-color: #0000FF;
        color: white;
    }
    .emotion-surprise {
        background-color: #FFFF00;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# ================== CACHE FUNCTIONS FOR PERFORMANCE ==================

@st.cache_resource
def load_bert_model():
    """Load BERT-BiLSTM model (cached)"""
    import os
    
    # Try both possible model filenames
    model_files = ['best_optimized_bert_bilstm_model.h5']
    model_path = None
    
    for filename in model_files:
        if os.path.exists(filename):
            model_path = filename
            break
    
    if model_path is None:
        st.error("❌ Model file not found!")
        st.info(f"📌 Looking for: {', '.join(model_files)}")
        st.info("📌 Make sure the model file is in the same directory as this app.")
        return None
    
    try:
        # Rebuild model architecture to match training script exactly
        from transformers import TFAutoModel
        from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense, BatchNormalization, Lambda
        from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
        from tensorflow.keras.models import Model
        
        with st.spinner("🔄 Loading BERT base model..."):
            bert_model = TFAutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True, from_pt=True)
            bert_config = bert_model.config
        
        # Recreate the exact architecture from training script (script.py)
        max_len = 128
        num_emotion_classes = 7
        num_sentiment_classes = 3
        
        # Input layers (3 inputs to match training)
        input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
        token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
        
        # BERT Lambda layer (recreate call_bert function to match training)
        def call_bert(inputs):
            ids, mask, token_ids = inputs
            return bert_model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids)[0]
        
        sequence_output = Lambda(
            call_bert,
            output_shape=(max_len, bert_config.hidden_size),
            name='bert_lambda'
        )([input_ids, attention_mask, token_type_ids])
        
        # BiLSTM layers (matching training architecture)
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
        
        # Global average and max pooling
        bilstm_avg = GlobalAveragePooling1D()(bilstm)
        bilstm_max = GlobalMaxPooling1D()(bilstm)
        bilstm_final = Concatenate()([bilstm_avg, bilstm_max])
        
        # Shared dense layers (matching training)
        shared = Dense(256, activation='relu', kernel_initializer='he_normal', name='shared_dense_1')(bilstm_final)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3, name='shared_dropout_1')(shared)
        
        shared = Dense(128, activation='relu', kernel_initializer='he_normal', name='shared_dense_2')(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3, name='shared_dropout_2')(shared)
        
        shared = Dense(64, activation='relu', kernel_initializer='he_normal', name='shared_dense_3')(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.3, name='shared_dropout_3')(shared)
        
        # Emotion head (matching training)
        emotion_dense = Dense(64, activation='relu', kernel_initializer='he_normal', name='emotion_dense_1')(shared)
        emotion_dense = BatchNormalization()(emotion_dense)
        emotion_dense = Dropout(0.3, name='emotion_dropout_1')(emotion_dense)
        
        emotion_dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='emotion_dense_2')(emotion_dense)
        emotion_dense = Dropout(0.2, name='emotion_dropout_2')(emotion_dense)
        
        emotion_output = Dense(num_emotion_classes, activation='softmax', name='emotion_output')(emotion_dense)
        
        # Sentiment head (matching training)
        sentiment_dense = Dense(64, activation='relu', kernel_initializer='he_normal', name='sentiment_dense_1')(shared)
        sentiment_dense = BatchNormalization()(sentiment_dense)
        sentiment_dense = Dropout(0.3, name='sentiment_dropout_1')(sentiment_dense)
        
        sentiment_dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='sentiment_dense_2')(sentiment_dense)
        sentiment_dense = Dropout(0.2, name='sentiment_dropout_2')(sentiment_dense)
        
        sentiment_output = Dense(num_sentiment_classes, activation='softmax', name='sentiment_output')(sentiment_dense)
        
        # Create model (3 inputs to match training)
        model = Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=[emotion_output, sentiment_output],
            name='BERT_BiLSTM_MultiTask'
        )
        
        # Load weights from saved model (always rebuild architecture to avoid Lambda serialization issues)
        with st.spinner("🔄 Loading model weights..."):
            try:
                custom_objects = {'Lambda': Lambda, 'call_bert': call_bert}
                temp_model = load_model(model_path, custom_objects=custom_objects, compile=False)
                
                # Copy weights from loaded model to rebuilt model (layer by layer)
                weights_loaded = 0
                for layer in model.layers:
                    try:
                        temp_layer = temp_model.get_layer(layer.name)
                        if temp_layer is not None:
                            weights = temp_layer.get_weights()
                            if weights:
                                layer.set_weights(weights)
                                weights_loaded += 1
                    except (ValueError, AttributeError) as e:
                        # Skip layers that don't have weights or don't match
                        continue
                
                # Clean up temporary model
                del temp_model
                st.success(f"✅ Loaded weights for {weights_loaded} layers")
                
            except Exception as weights_error:
                st.error(f"Could not load weights: {weights_error}")
                st.info("💡 The model file might be corrupted or incompatible.")
                raise weights_error
        
        # Compile model after loading weights
        model.compile(
            loss=['categorical_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.7, 0.3],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
            metrics=[['accuracy'], ['accuracy']]
        )
        
        st.success(f"✅ Model loaded successfully: {model_path}")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        st.warning("💡 **Troubleshooting:**")
        st.info("""
        1. Make sure transformers, tf-keras, and torch are installed:
           `pip install transformers tf-keras torch`
        2. The model architecture is being rebuilt to avoid serialization issues
        3. If this still fails, you may need to retrain and save the model differently
        """)
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
        st.info("💡 Make sure transformers library is installed: pip install transformers")
        return None

# ================== PREDICTION FUNCTION ==================

def predict_emotion_sentiment(text, model, tokenizer, max_len=128):
    """Predict emotion and sentiment for text"""
    
    if not text.strip():
        return None
    
    try:
        # Tokenize (include token_type_ids to match model architecture)
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
        
        # Predict (3 inputs to match model architecture)
        emotion_pred, sentiment_pred = model.predict(
            [input_ids, attention_mask, token_type_ids],
            verbose=0
        )
        
        # Process emotion predictions (model.predict returns batch, so get first item)
        emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        emotion_pred = emotion_pred[0]  # Get first batch item
        emotion_idx = np.argmax(emotion_pred)
        emotion = emotion_labels[emotion_idx]
        emotion_confidence = emotion_pred[emotion_idx]
        emotion_probs = {emotion_labels[i]: float(emotion_pred[i]) for i in range(len(emotion_labels))}
        
        # Process sentiment predictions
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment_pred = sentiment_pred[0]  # Get first batch item
        sentiment_idx = np.argmax(sentiment_pred)
        sentiment = sentiment_labels[sentiment_idx]
        sentiment_confidence = sentiment_pred[sentiment_idx]
        sentiment_probs = {sentiment_labels[i]: float(sentiment_pred[i]) for i in range(len(sentiment_labels))}
        
        # Emotion to color mapping
        emotion_colors = {
            'anger': '#FF0000',
            'disgust': '#8B008B',
            'fear': '#FFA500',
            'joy': '#00FF00',
            'neutral': '#808080',
            'sadness': '#0000FF',
            'surprise': '#FFFF00'
        }
        
        # Sentiment to emoji mapping
        sentiment_emojis = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }
        
        return {
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'emotion_probs': emotion_probs,
            'emotion_color': emotion_colors[emotion],
            'sentiment': sentiment,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_probs': sentiment_probs,
            'sentiment_emoji': sentiment_emojis[sentiment]
        }
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def color_sentences(text, emotion_probs, sentiment_probs, emotion, sentiment):
    """Create colored HTML for sentences"""
    
    emotion_colors = {
        'anger': '#FF0000',
        'disgust': '#8B008B',
        'fear': '#FFA500',
        'joy': '#00FF00',
        'neutral': '#808080',
        'sadness': '#0000FF',
        'surprise': '#FFFF00'
    }
    
    sentiment_emojis = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '😞'
    }
    
    color = emotion_colors[emotion]
    emoji = sentiment_emojis[sentiment]
    
    html = f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: white; font-weight: bold; text-align: center; font-size: 18px;">
        {text} {emoji}
    </div>
    """
    
    return html

# ================== MAIN APP ==================

def main():
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🎭 BERT-BiLSTM Emotion Detection</h1>
        <p>Analyze emotions and sentiments in text using advanced Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    mode = st.sidebar.radio(
        "Choose Mode:",
        ["Single Text", "Multiple Texts", "Example Sentences"]
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("📊 About Model")
    st.sidebar.info("""
    **Model:** BERT-BiLSTM Hybrid
    
    **Features:**
    - 7 Emotion Classes
    - 3 Sentiment Classes
    - Contextual Understanding
    - Multi-task Learning
    
    **Accuracy:** 80-88%
    """)
    
    # Load model and tokenizer with loading indicators
    with st.spinner("🔄 Loading model and tokenizer..."):
        model = load_bert_model()
        tokenizer = load_bert_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("⚠️ Failed to load model or tokenizer. Please check your setup.")
        st.stop()  # Stop execution instead of return
    
    # ================== MODE 1: SINGLE TEXT ==================
    
    if mode == "Single Text":
        st.header("🔍 Analyze Single Text")
        
        text_input = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="Type something like: 'I'm so happy!' or 'This is terrible!'"
        )
        
        if st.button("🚀 Analyze Emotion", key="single_analyze"):
            if text_input.strip():
                result = predict_emotion_sentiment(text_input, model, tokenizer)
                
                if result:
                    # Display colored text
                    st.subheader("🎨 Colored Output")
                    colored_html = color_sentences(
                        text_input,
                        result['emotion_probs'],
                        result['sentiment_probs'],
                        result['emotion'],
                        result['sentiment']
                    )
                    st.markdown(colored_html, unsafe_allow_html=True)
                    
                    # Results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("😊 Emotion Result")
                        st.success(f"**Emotion:** {result['emotion'].upper()}")
                        st.metric(
                            "Confidence",
                            f"{result['emotion_confidence']:.2%}"
                        )
                        
                        # Emotion distribution chart
                        emotion_df = pd.DataFrame(
                            list(result['emotion_probs'].items()),
                            columns=['Emotion', 'Probability']
                        ).sort_values('Probability', ascending=False)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=emotion_df['Emotion'],
                                y=emotion_df['Probability'],
                                marker_color=['#FF0000', '#8B008B', '#FFA500', '#00FF00', '#808080', '#0000FF', '#FFFF00']
                            )
                        ])
                        fig.update_layout(
                            title="Emotion Distribution",
                            xaxis_title="Emotion",
                            yaxis_title="Probability",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("💭 Sentiment Result")
                        
                        sentiment_color_map = {
                            'positive': '🟢',
                            'neutral': '🟡',
                            'negative': '🔴'
                        }
                        
                        st.success(f"**Sentiment:** {sentiment_color_map[result['sentiment']]} {result['sentiment'].upper()}")
                        st.metric(
                            "Confidence",
                            f"{result['sentiment_confidence']:.2%}"
                        )
                        
                        # Sentiment distribution chart
                        sentiment_df = pd.DataFrame(
                            list(result['sentiment_probs'].items()),
                            columns=['Sentiment', 'Probability']
                        )
                        
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=sentiment_df['Sentiment'],
                                values=sentiment_df['Probability'],
                                marker=dict(colors=['#0000FF', '#808080', '#00FF00'])
                            )
                        ])
                        fig.update_layout(
                            title="Sentiment Distribution",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.subheader("📋 Detailed Probabilities")
                    
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.write("**Emotion Probabilities:**")
                        emotion_df_detailed = pd.DataFrame(
                            list(result['emotion_probs'].items()),
                            columns=['Emotion', 'Probability']
                        ).sort_values('Probability', ascending=False)
                        emotion_df_detailed['Probability'] = emotion_df_detailed['Probability'].apply(lambda x: f"{x:.2%}")
                        st.table(emotion_df_detailed)
                    
                    with prob_col2:
                        st.write("**Sentiment Probabilities:**")
                        sentiment_df_detailed = pd.DataFrame(
                            list(result['sentiment_probs'].items()),
                            columns=['Sentiment', 'Probability']
                        ).sort_values('Probability', ascending=False)
                        sentiment_df_detailed['Probability'] = sentiment_df_detailed['Probability'].apply(lambda x: f"{x:.2%}")
                        st.table(sentiment_df_detailed)
            else:
                st.warning("⚠️ Please enter some text!")
    
    # ================== MODE 2: MULTIPLE TEXTS ==================
    
    elif mode == "Multiple Texts":
        st.header("📝 Analyze Multiple Texts")
        
        st.info("Enter multiple sentences separated by a newline. Each will be analyzed separately.")
        
        multi_text = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="I'm so happy!\nThis is terrible!\nI'm not sure..."
        )
        
        if st.button("🚀 Analyze All", key="multi_analyze"):
            texts = [t.strip() for t in multi_text.split('\n') if t.strip()]
            
            if texts:
                results = []
                
                for i, text in enumerate(texts, 1):
                    result = predict_emotion_sentiment(text, model, tokenizer)
                    if result:
                        results.append({
                            'Text': text,
                            'Emotion': result['emotion'],
                            'Emotion Conf': f"{result['emotion_confidence']:.2%}",
                            'Sentiment': result['sentiment'],
                            'Sentiment Conf': f"{result['sentiment_confidence']:.2%}"
                        })
                
                # Display results table
                st.subheader("📊 Results Summary")
                results_df = pd.DataFrame(results)
                st.table(results_df)
                
                # Display each with coloring
                st.subheader("🎨 Colored Outputs")
                for i, (text, result) in enumerate(zip(texts, results), 1):
                    st.write(f"**{i}. {text}**")
                    pred = predict_emotion_sentiment(text, model, tokenizer)
                    colored_html = color_sentences(
                        text,
                        pred['emotion_probs'],
                        pred['sentiment_probs'],
                        pred['emotion'],
                        pred['sentiment']
                    )
                    st.markdown(colored_html, unsafe_allow_html=True)
                    st.divider()
            else:
                st.warning("⚠️ Please enter at least one text!")
    
    # ================== MODE 3: EXAMPLES ==================
    
    elif mode == "Example Sentences":
        st.header("📚 Example Sentences")
        
        examples = [
            "I'm so happy! This is amazing!",
            "I'm really angry about this!",
            "This is absolutely disgusting!",
            "I'm terrified and scared!",
            "I was sad but now I'm happy!",
            "Wow! That's so surprising!",
            "The meeting is at 3 PM.",
        ]
        
        st.info(f"Showing {len(examples)} example sentences. Click any to analyze!")
        
        selected_example = st.selectbox(
            "Choose an example:",
            examples
        )
        
        if st.button("🚀 Analyze Example", key="example_analyze"):
            result = predict_emotion_sentiment(selected_example, model, tokenizer)
            
            if result:
                # Display colored text
                st.subheader("🎨 Result")
                colored_html = color_sentences(
                    selected_example,
                    result['emotion_probs'],
                    result['sentiment_probs'],
                    result['emotion'],
                    result['sentiment']
                )
                st.markdown(colored_html, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Emotion", result['emotion'].upper())
                
                with col2:
                    st.metric("Emotion Confidence", f"{result['emotion_confidence']:.2%}")
                
                with col3:
                    st.metric("Sentiment", result['sentiment'].upper())
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    emotion_df = pd.DataFrame(
                        list(result['emotion_probs'].items()),
                        columns=['Emotion', 'Probability']
                    ).sort_values('Probability', ascending=False)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=emotion_df['Emotion'],
                            y=emotion_df['Probability'],
                            marker_color=['#FF0000', '#8B008B', '#FFA500', '#00FF00', '#808080', '#0000FF', '#FFFF00']
                        )
                    ])
                    fig.update_layout(title="Emotion Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    sentiment_df = pd.DataFrame(
                        list(result['sentiment_probs'].items()),
                        columns=['Sentiment', 'Probability']
                    )
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=sentiment_df['Sentiment'],
                            values=sentiment_df['Probability'],
                            marker=dict(colors=['#0000FF', '#808080', '#00FF00'])
                        )
                    ])
                    fig.update_layout(title="Sentiment Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()