import streamlit as st
import pandas as pd
import numpy as np
import nltk
from os import path
import io
from PIL import Image
import altair as alt
from nltk import WordNetLemmatizer, SnowballStemmer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score, recall_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings


warnings.filterwarnings('ignore')

nltk.download('wordnet')
nltk.download('stopwords')

def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, sep="\t")
        return data
    return None

def preprocess_data(data, text_col):
    data_copy = data.copy()
    if data_copy[text_col].dtype == 'O':
        data_copy[text_col] = data_copy[text_col].str.lower()
        data_copy[text_col] = data_copy[text_col].str.replace('[^\w\s]', '', regex=True)
        data_copy[text_col] = data_copy[text_col].str.replace('\d', '', regex=True)
        sw = set(stopwords.words('english'))
        data_copy[text_col] = data_copy[text_col].apply(lambda x: " ".join(word for word in x.split() if word not in sw))
        # Köklerin alınması (Opsiyonel)
        lemmatizer = WordNetLemmatizer()
        data_copy[text_col] = data_copy[text_col].apply(
            lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))

        # Stemming (Opsiyonel)
        stemmer = SnowballStemmer("english")
        data_copy[text_col] = data_copy[text_col].apply(lambda x: " ".join(stemmer.stem(word) for word in x.split()))

    else:
        st.warning(f"Column '{text_col}' is not an object type.")
    return data_copy


def vectorize_text(vectorizer_name, X_train, X_test=None, ngram_range=(1, 1)):
    if vectorizer_name == "TF-IDF":
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test) if X_test is not None and vectorizer is not None else None
        return X_train_vectorized, X_test_vectorized, vectorizer
    elif vectorizer_name == "Count Vectorizer":
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test) if X_test is not None and vectorizer is not None else None
        return X_train_vectorized, X_test_vectorized, vectorizer
    else:
        st.write("Invalid vectorizer name.")
        return None, None, None


def train_model(X_train, y_train, model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000)
    elif model_name == "Naive Bayes":
        model = MultinomialNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Support Vector Machine":
        model = SVC(probability=True)

    model.fit(X_train, y_train)
    return model

def sentiment_analysis(data, text_col, sentiment_col, vectorizer_name, model_name, ngram_range=(1, 1)):
    """
    Performs sentiment analysis on the given data using the specified vectorizer and model.

    Args:
    - data (DataFrame): The input data containing text and sentiment columns.
    - text_col (str): The name of the column containing the text data.
    - sentiment_col (str): The name of the column containing the sentiment labels.
    - vectorizer_name (str): The name of the text vectorizer to use ('TF-IDF' or 'Count Vectorizer').
    - model_name (str): The name of the machine learning model to use for sentiment analysis.
    - ngram_range (tuple): The range of n-grams to use for vectorization (default is (1, 1)).

    Returns:
    - trained_model (object): The trained machine learning model.
    - vectorizer (object): The text vectorizer used for training the model.
    - label_encoder (object): The label encoder used for training the model.
    - accuracy (float): The accuracy score of the trained model.
    - report (str): The classification report of the trained model.
    """
    if not data.empty and sentiment_col in data.columns:
        # Encode the sentiment labels
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data[sentiment_col])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data[text_col], data['label'], test_size=0.2, random_state=42)

        # Vectorize the text data
        X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(vectorizer_name, X_train, X_test, ngram_range)

        # Train the model
        trained_model = train_model(X_train_vectorized, y_train, model_name)

        # Evaluate the model
        accuracy, report = evaluate_model(trained_model, vectorizer, X_test_vectorized, y_test,data)

        return trained_model, vectorizer, label_encoder, accuracy, report
    else:
        st.write("Please upload a non-empty dataset with a 'Sentiment' column for sentiment analysis.")
        return None, None, None, None, None

def evaluate_model(model, vectorizer, X_test, y_test, data):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    # Gerçek etiketlere eriş
    true_labels = y_test  # Değişiklik burada

    # Gerçek etiketler kullanılarak değerlendirme yap
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    class_labels = data['label'].unique()

    # Görselleştirme
    plot_confusion_matrix_with_metrics(cm, class_labels,fp, fn, tp, tn)  # Etiketlerinizi uygun şekilde değiştirin

    # Metriklerin hesaplanması
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1_score = 2 * (precision * recall) / (precision + recall)
    display_metrics(precision, recall, accuracy, f1_score)
    return report, accuracy

def plot_confusion_matrix_with_metrics(cm, labels, fp, fn, tp, tn):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)  # Yazı boyutunu ayarla
    sns.heatmap(cm, annot=True, cmap='viridis', fmt='g', xticklabels=labels, yticklabels=labels,annot_kws={"size": 14})  # Renkler, formatlama ve etiketler
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')  # Etiketleri döndür ve sağa hizala
    plt.yticks(rotation=45, ha='right')  # Etiketleri döndür ve sağa hizala

    # True Positive değeri kutu üzerine yazdırma
    plt.text(0.5, 0.5, f'TP\n{tp}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5), fontsize=12, fontweight='bold', color='black')
    # True Negative değeri kutu üzerine yazdırma
    plt.text(1.5, 0.5, f'TN\n{tn}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5), fontsize=12, fontweight='bold', color='black')
    # False Positive değeri kutu üzerine yazdırma
    plt.text(0.5, 1.5, f'FP\n{fp}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5), fontsize=12, fontweight='bold', color='black')
    # False Negative değeri kutu üzerine yazdırma
    plt.text(1.5, 1.5, f'FN\n{fn}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5), fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()  # Grafik düzenini ayarla
    st.pyplot(plt)


    # Confusion matrix yorumunu ekle
    interpretation = interpret_confusion_matrix(fp, fn, tp, tn)
    st.write("Confusion Matrix Interpretation:")
    st.write(interpretation)

    # Expander
    with st.expander("Confusion Matrix Interpretation Details"):
        st.write("This section provides details about the interpretation of the confusion matrix.")
        st.write(interpretation)
        st.write("You can add more information here if needed.")



def display_metrics(precision, recall, accuracy, f1_score):
    st.write("### Evaluation Metrics:")

    # Precision
    st.write(f"Precision: {precision:.2f}")
    precision_widget = st.progress(precision)
    precision_info = st.info("Precision tells us what proportion of positive identifications was actually correct.")

    # Recall
    st.write(f"Recall: {recall:.2f}")
    recall_widget = st.progress(recall)
    recall_info = st.info("Recall tells us what proportion of actual positives was identified correctly.")

    # Accuracy
    st.write(f"Accuracy: {accuracy:.2f}")
    accuracy_widget = st.progress(accuracy)
    accuracy_info = st.info("Accuracy tells us the overall proportion of correct predictions.")

    # F1 Score
    st.write(f"F1 Score: {f1_score:.2f}")
    f1_score_widget = st.progress(f1_score)
    st.info("F1 Score is the harmonic mean of precision and recall. It's a balanced performance measure.")



def analyze_sentiment(text, method, trained_model=None, vectorizer=None, label_encoder=None):
    if text.strip() == "":
        return None, None, None, None  # Hata durumunu işaretleyerek None değerleri döndür

    # Ön işlem adımları
    processed_text = text.lower()  # Metni küçük harfe dönüştür
    processed_text = re.sub(r'[^\w\s]', '', processed_text)  # Gereksiz karakterleri kaldır
    processed_text = re.sub(r'\d', '', processed_text)  # Sayıları kaldır

    # Stop words kaldırma
    sw = set(stopwords.words('english'))
    processed_text = " ".join(word for word in processed_text.split() if word not in sw)

    # Köklerin alınması (Opsiyonel)
    lemmatizer = WordNetLemmatizer()
    processed_text = " ".join(lemmatizer.lemmatize(word) for word in processed_text.split())

    # Stemming (Opsiyonel)
    stemmer = SnowballStemmer("english")
    processed_text = " ".join(stemmer.stem(word) for word in processed_text.split())

    # Vectorize the input text
    if isinstance(vectorizer, (TfidfVectorizer, CountVectorizer)):
        X_vectorized = vectorizer.transform([processed_text])
    else:
        st.error("Geçersiz vektörleştirici türü.")
        return None, None, None, None  # Hata durumunu işaretleyerek None değerleri döndür

    # Tahmini yap
    prediction = trained_model.predict(X_vectorized)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    if predicted_label == "yes":
        sentiment_score = 1
    elif predicted_label == "no":
        sentiment_score = -1
    else:
        sentiment_score = 0  # Tahmin edilen etiket "Neutral" ise duygu skorunu 0 olarak ayarla

    if method == 'VADER':
        # VADER kullanarak duygu analizi yap
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(processed_text)
        compound_score = scores['compound']

        positive_threshold = 0.1  # Pozitif eşik değeri
        negative_threshold = -0.1  # Negatif eşik değeri

        # Overall Sentiment'i belirle
        if compound_score >= positive_threshold:
            overall_sentiment = "Positive"
        elif compound_score <= negative_threshold:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        return compound_score, scores, overall_sentiment, predicted_label, sentiment_score

    elif method == 'TextBlob':
        # TextBlob kullanarak duygu analizi yap
        blob = TextBlob(processed_text)
        sentiment = blob.sentiment

        # Compound score'u TextBlob sonucuna göre ayarla
        compound_score = sentiment.polarity

        # Overall Sentiment'i belirle
        if compound_score > 0:
            overall_sentiment = "Positive"
        elif compound_score < 0:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        # Diğer duygu skorlarını sıfır olarak ayarla (TextBlob bunları sağlamıyor)
        scores = {'neg': 0, 'neu': 0, 'pos': 0}

        return compound_score, scores, overall_sentiment, predicted_label, sentiment_score

    elif method == 'Machine Learning':
        # Makine öğrenmesi yöntemini kullanarak duygu analizi yap
        if isinstance(trained_model, (LogisticRegression, MultinomialNB, RandomForestClassifier, SVC)):
            # Tahmin etmek için vectorized metni kullan
            prediction_proba = trained_model.predict_proba(X_vectorized)[0]
            predicted_class = trained_model.predict(X_vectorized)[0]

            if predicted_class == 1:
                predicted_label = "yes"
            else:
                predicted_label = "no"

            # Tahmin olasılıklarını döndür
            return prediction_proba, None, None, predicted_label, sentiment_score
        else:
            st.error("Geçersiz makine öğrenmesi modeli.")
            return None, None, None, None, None

    else:
        st.error("Geçersiz yöntem. Lütfen 'VADER', 'TextBlob' veya 'Machine Learning' seçin.")
        return None, None, None, None, None  # Hata durumunu işaretleyerek None değerleri döndür


def get_emoji_for_sentiment(predicted_label):
    sentiment_emojis = {
        "positive": "😄",
        "neutral": "😐",
        "negative": "😡"
    }

    return sentiment_emojis.get(predicted_label, "😕")

def visualize_sentiment_radar(prediction_proba, predicted_label, sentiment_score):

    probability_yes = prediction_proba[1] * 100
    probability_no = prediction_proba[0] * 100
    st.markdown(f"""
        ### Prediction Probabilities:
        - **Positive (Yes)**: {probability_yes:.1f}%
        - **Negative (No)**: {probability_no:.1f}%
        """)
    data = pd.DataFrame({
        'Sentiment': ['Positive (Yes)', 'Negative (No)'],
        'Probability': [probability_yes, probability_no]
    })

    color_scale = alt.Scale(
        domain=['Positive (Yes)', 'Negative (No)'],
        range=['green', 'red']
    )

    chart = alt.Chart(data).mark_bar().encode(
        x='Sentiment',
        y=alt.Y('Probability', axis=alt.Axis(title='Probability (%)'), scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Sentiment', scale=color_scale)
    ).properties(
        width=400,
        height=300
    )

    text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-10,
        color='black'
    ).encode(
        text=alt.Text('Probability', format='.1f')
    )

    st.altair_chart(chart + text)


def plot_compound_score_distribution(data, text_col):
    if 'compound' in data.columns:
        analyzer = SentimentIntensityAnalyzer()
        data['compound'] = data[text_col].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        plt.figure(figsize=(10, 6))
        plt.hist(data['compound'], bins=20, edgecolor='black')
        plt.title('Compound Score Distribution')
        plt.xlabel('Compound Score')
        plt.ylabel('Frequency')
        st.pyplot(plt)
    else:
        st.warning("The 'compound' column does not exist in the dataset.")

def create_wordcloud(data, text_col):
    text = " ".join(review for review in data[text_col])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


def load_image(image_file):
    if image_file is not None:
        try:
            # Resmi aç
            image = Image.open(io.BytesIO(image_file.read()))
            return image
        except Exception as e:
            st.error(f"An error occurred while loading the image: {str(e)}")
            return None
    else:
        st.warning("Please upload an image.")
        return None


def generate_wordcloud_with_mask(data, text_col, mask_image_file):
    mask_image = load_image(mask_image_file)
    if mask_image is not None:
        mask = np.array(mask_image.convert('L'))  # Grayscale conversion for better mask processing
        wc = WordCloud(
            background_color="grey",
            mode="RGB",
            max_words=200,
            mask=mask,
            contour_width=3,
            contour_color="firebrick",
            max_font_size=100,
            min_font_size=10,
            random_state=42,
            colormap="viridis"  # Adjust the color map for better visualization
        ).generate(" ".join(data[text_col]))

        plt.figure(figsize=[10, 10])
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.warning("Failed to load the image.")

def format_classification_report(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = line.split()
        class_label = row_data[0]
        precision = float(row_data[1])
        recall = float(row_data[2])
        f1_score = float(row_data[3])
        support = int(row_data[4])
        row['Class'] = class_label
        row['Precision'] = precision
        row['Recall'] = recall
        row['F1-Score'] = f1_score
        row['Support'] = support
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    formatted_report = df.to_markdown(index=False)
    return formatted_report
#option = st.sidebar.selectbox("Menü", ["Sentiment Analysis", "WordCloud"])
    #if option == "Sentiment Analysis":
        #st.sidebar.markdown("<span style='color:yellow'>Sentiment Analysis</span>", unsafe_allow_html=True)
    #elif option == "WordCloud":
        #st.sidebar.markdown("<span style='color:yellow'>WordCloud</span>", unsafe_allow_html=True)
def interpret_confusion_matrix(fp, fn, tp, tn):
    interpretation = ""

    # If True Positive and True Negative are high, indicate that the model performs well overall
    if tp > 0.8 * (tp + fn) and tn > 0.8 * (tn + fp):
        interpretation = "The model demonstrates good overall performance. This indicates that the model correctly predicts both positive and negative instances with high accuracy."
    # If False Positive and False Negative are high, suggest that the model needs improvement
    elif fp > 0.2 * (fp + tn) and fn > 0.2 * (fn + tp):
        interpretation = "The model needs improvement. Particularly, efforts should be made to reduce false positive and false negative results. This suggests that the model is misclassifying both positive and negative instances frequently."
    # In other cases, provide a general comment
    else:
        interpretation = "A comment about the model's performance could not be made. Further analysis may be required. This suggests that the model's performance is not clearly defined based on the confusion matrix alone."
    # Markdown içinde CSS kullanarak yazı rengini değiştirme

    return interpretation


def main():
    st.title("Sentiment Analysis and WordCloud Application")


    if 'data' not in st.session_state:
        st.session_state['data'] = None
    st.sidebar.markdown(
        f'<h3 style="color:yellow;">Upload File</h3>',
        unsafe_allow_html=True
    )
    uploaded_file = st.sidebar.file_uploader("", type=["csv", "tsv"])
    if uploaded_file:
        st.write("")

        st.session_state['data'] = load_data(uploaded_file)
        st.write("")
        st.write("")
        st.write("")

        if 'data' in st.session_state and st.session_state['data'] is not None:
            st.success("Data loaded and preprocessed successfully.")
            st.write(" Summary of the loaded data")
            st.dataframe(st.session_state['data'].head())
            st.info(f"Shape of data: {st.session_state['data'].shape}")
            st.session_state['text_col'] = st.selectbox("Select the text column for analysis",
                                                        st.session_state['data'].columns)
            st.write("")
            st.write("")

            st.session_state['sentiment_col'] = st.selectbox("Select a column for Sentiment analysis:",
                                                             st.session_state['data'].columns)
            st.write("")
            st.session_state['data'] = preprocess_data(st.session_state['data'], st.session_state['text_col'])

    #st.sidebar.markdown("<h4 style='color: yellow;'>MENU</h4>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<h3 style="color:yellow;">Menu</h3>',
        unsafe_allow_html=True
    )
    option = st.sidebar.selectbox("", ["Sentiment Analysis", "WordCloud"])
    st.sidebar.text("")
    if st.session_state['data'] is not None:
        if option == "Sentiment Analysis":
            st.sidebar.markdown(
                f'<h3 style="color:yellow;">Pre-process</h3>',
                unsafe_allow_html=True
            )

                  # Sidebar'da vectorizer seçeneklerini göster
            st.sidebar.text("Select Vectorizer")
            vectorizer_name = st.sidebar.radio("", ["TF-IDF", "Count Vectorizer"])

            # TF-IDF veya Count Vectorizer seçilmişse, n-gram aralığını seçme düğmelerini göster
            if vectorizer_name in ["TF-IDF", "Count Vectorizer"]:
                st.sidebar.text("Select N-gram Range:")

                # Min ve Max n-gram değerlerini tanımla
                ngram_min = st.sidebar.number_input("Min", min_value=1, max_value=5, value=1)
                ngram_max = st.sidebar.number_input("Max", min_value=ngram_min, max_value=5, value=ngram_min)

                ngram_range = (ngram_min, ngram_max)
            else:
                ngram_range = (1, 1)  # Diğer vectorizer seçenekleri için varsayılan olarak 1-gram aralığı kullan
            st.sidebar.markdown("")
            st.sidebar.markdown("")
            st.sidebar.markdown(
                f'<h3 style="color:yellow;">Model Training</h3>',
                unsafe_allow_html=True
            )

            st.sidebar.text("Select Model")
            model_name = st.sidebar.selectbox("", ["Logistic Regression", "Naive Bayes", "Random Forest", "Support Vector Machine"])

            if st.sidebar.button("Train Model"):
                with st.spinner("Training the model..."):
                    trained_model, vectorizer, label_encoder, accuracy, report = sentiment_analysis(
                        st.session_state['data'], st.session_state['text_col'],st.session_state['sentiment_col'], vectorizer_name, model_name, ngram_range
                    )
                    st.session_state['trained_model'] = trained_model
                    st.session_state['vectorizer'] = vectorizer
                    st.session_state['label_encoder'] = label_encoder
                    st.success("Training completed.")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.session_state['accuracy'] = accuracy
                    st.session_state['report'] = report
                    st.write("")
                    st.write("")
                    st.write("")

                    if report is not None:



                       # accuracy_percentage = float(st.session_state['accuracy']) * 100
                        #st.write(f"Accuracy: {accuracy_percentage:.2f}%")

                        #progress_bar = st.progress(round(accuracy_percentage))

                        st.write("")

                        #st.write("Classification Report:")
                        #formatted_report = format_classification_report(st.session_state['report'])
                        #st.write(formatted_report)
                    else:
                        st.warning("Classification report is missing. Please check your training process.")

            if 'trained_model' in st.session_state and st.session_state['trained_model'] is not None:
                st.write("")
                st.write("")
                st.write("")
                st.write("")

                #method = st.radio("Select sentiment analysis method:", ('VADER', 'TextBlob'))
                st.subheader("Predict a new comment")
                text_input = st.text_input("Enter text for sentiment prediction:")
                if st.button("Predict"):
                    if text_input:
                        st.markdown(f"<span style='color: grey;'>{text_input}</span>", unsafe_allow_html=True)

                        # Machine Learning method
                        prediction_proba_ml, _, _, predicted_label_ml, sentiment_score_ml = analyze_sentiment(
                            text_input, 'Machine Learning', st.session_state['trained_model'],
                            st.session_state['vectorizer'], st.session_state['label_encoder'])

                        if prediction_proba_ml is not None:
                            st.write("")
                            st.write("")
                            st.write("")
                            visualize_sentiment_radar(prediction_proba_ml, predicted_label_ml, sentiment_score_ml)
                        else:
                            st.error("Error in analyzing sentiment.")
                    else:
                        st.warning("Please enter some text for sentiment prediction.")

            else:
                st.warning("Model training failed. Please check your data and parameters.")
            if st.button("Clear Results"):
                st.session_state.pop('trained_model', None)
                st.session_state.pop('vectorizer', None)
                st.session_state.pop('label_encoder', None)
                st.session_state.pop('accuracy', None)
                st.session_state.pop('report', None)
                st.success("Results cleared.")
        elif option == "WordCloud":
            st.sidebar.markdown("<h4 style='color: yellow;'>Upload a mask image</h4>", unsafe_allow_html=True)
            mask_image_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])

            if st.sidebar.button("Generate WordCloud"):

                create_wordcloud(st.session_state['data'], st.session_state['text_col'])

                if mask_image_file:
                    st.image(mask_image_file)
                    try:
                        generate_wordcloud_with_mask(st.session_state['data'], st.session_state["text_col"],
                                                     mask_image_file)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")


    else:
        st.warning("Please upload a dataset to proceed.")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.info(
        "ℹ️ Use the sidebar to navigate between different options.\n"
        "1. **Sentiment Analysis:** Train models, predict sentiment, and view results.\n"
        "2. **WordCloud:** Generate WordCloud from text data.\n\n"
        "After selecting an option, follow the instructions to upload data, choose models, and perform actions."
    )
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.sidebar.info(st.info("Created by [Çiğdem Sıcakyüz]"))

if __name__ == "__main__":
    main()

