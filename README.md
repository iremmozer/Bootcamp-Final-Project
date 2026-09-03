# Bootcamp-Final-Project
## Model 1: Regression Analysis

We used the `proje_ev_fiyatlari.csv` dataset to build this model. First, we defined empty lists to store model names, MSE, and R² values for comparison in a final summary table.

Next, we defined the feature matrix ($X$) and target variable ($y$), then split the data into training and testing sets using `train_test_split` (`test_size=0.2`, `random_state=42`).

The pipeline included the following implementations:
1. Feature Standardization
2. Linear Regression
3. Ridge Regression (with $\alpha = 1.0$)
4. Principal Component Regression (PCR: Standardization, PCA, Linear Regression)
5. Partial Least Squares (PLS Regression)
6. Artificial Neural Network (ANN)

The MSE and R² scores calculated at each step were appended to our metrics lists.

**Results:**
The Artificial Neural Network achieved the lowest error score (MSE). Its multi-layer architecture and configuration enabled it to effectively capture complex, non-linear relationships within the data.

---

## Model 2: Clustering Analysis

We used the `proje_musteri_segmentasyonu.csv` dataset for this task, selecting `annual_income` and `spending_score` as features ($X$).

After standardizing $X$, we evaluated optimal cluster counts using the Elbow Method (WCSS/Inertia plot across various $k$ values). We determined $k=5$ as the optimal number of clusters based on the distinct inflection point on the curve.

We then fitted the KMeans algorithm with $k=5$, generated predictions, and identified cluster centroids. To interpret the clusters in their original feature space, we applied `inverse_transform` to the standardized values. Finally, we visualized the clusters alongside their centroids and computed the mean annual income and spending score for each group.

**Evaluation:**
The model achieved a Silhouette Score of **0.72**, indicating clear cluster separation and high-quality grouping.

---

## Model 3: Text Sentiment Analysis

We built an NLP sentiment classification model using the `IMDB Dataset.csv`. Target labels (`sentiment`) were mapped to binary values (`positive` $\rightarrow$ 1, `negative` $\rightarrow$ 0).

We initialized and fitted a `Tokenizer` on the raw review texts (`fit_on_texts`). Since review lengths varied, we applied `pad_sequences` using a predefined maximum sequence length to ensure uniform input dimensions.

The dataset was split into train and test partitions. We constructed a Recurrent Neural Network (RNN) consisting of:
* An `Embedding` layer
* A `SimpleRNN` layer
* A `Dense` output layer with a sigmoid activation function

The network was trained by setting the batch size, number of epochs, and validation split. Evaluating on the test set yielded approximately **84% accuracy**, confirming reliable sentiment prediction.

Note on Dataset: Due to GitHub's file size limits, the dataset is not hosted directly in this repository. You can download the original dataset from Kaggle - IMDB Dataset of 50K Movie Reviews.

---

## Model 4: Image Classification

We utilized the Fashion-MNIST dataset imported directly from Keras/TensorFlow. The dataset was split into training and test sets (`X_train`, `y_train`, `X_test`, `y_test`). We reshaped the image arrays to include a single grayscale channel dimension and normalized pixel intensities to the $[0, 1]$ range by dividing by 255.

We built a Convolutional Neural Network (CNN) using the Sequential API with:
* `Conv2D` layers (ReLU activation)
* `MaxPooling2D` layers for spatial downsampling
* A `Flatten` layer to transition into fully connected layers
* `Dense` layers, ending with a 10-class `Softmax` output layer

The model was compiled and trained with specified epochs, batch size, and validation data. To diagnose misclassifications, we tracked error patterns across class names using a loop over `yanlis_tahmin` to identify which categories were most frequently confused.

**Results:**
The model performed well on unseen test data, achieving an accuracy of approximately **88%** with a test loss of **0.33**.
