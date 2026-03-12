# Customer Segmentation and Market Basket Analysis

## Project Overview

This project analyzes retail purchasing data to understand **who the customers are** and **how they buy products**.

The analysis combines two approaches:

1. **Customer segmentation** to identify groups of similar shoppers
2. **Purchase pattern analysis** to discover products that are often bought together

The goal is to generate insights that could help a retail business improve **targeted marketing, product recommendations, and cross‑selling strategies**.

---

## Dataset

The analysis uses a retail dataset containing **~550,000 purchase records**.

Each row represents a product purchased by a customer.

Main variables:

* **User_ID** – unique customer identifier
* **Product_ID** – purchased product
* **Gender** – customer gender
* **Age** – customer age group
* **Occupation** – occupation category
* **City_Category** – type of city where the customer lives
* **Marital_Status** – marital status
* **Product_Category** – category of the product
* **Purchase** – purchase amount

This dataset allows the analysis of both **customer demographics** and **purchasing behavior**.

---

## Objective

The objective of the project is to answer two practical questions:

1. **Are there different types of customers with distinct characteristics?**
2. **Are there products that customers tend to buy together?**

Answering these questions helps retailers better understand their customer base and improve sales strategies.

---

## Approach

### 1. Customer Segmentation

Customers are grouped using the **K‑Means clustering algorithm** based on:

* Age
* Gender
* Marital status
* City category

To determine the appropriate number of clusters, several evaluation metrics are used:

* Elbow Method
* Silhouette Score
* Davies–Bouldin Score
* Calinski–Harabasz Score

The final model produces **distinct customer segments**, each representing a different demographic profile.

Examples of possible segments include:

* Younger single shoppers
* Middle‑aged married customers
* Customers from large metropolitan areas

These segments can help businesses design **more targeted campaigns and promotions**.

---

### 2. Purchase Pattern Analysis

After identifying customer segments, the project analyzes **products frequently purchased by the same customers**.

This step identifies combinations of products that often appear together in customer purchases.

These patterns can be useful for:

* Product recommendations
* Bundling strategies
* Cross‑selling

The analysis is applied within a selected customer segment to better understand the purchasing behavior of that group.

---

## Results

The project produces several outputs:

**Customer segmentation insights**

* Size of each customer segment
* Demographic profile of each segment

**Cluster evaluation metrics**

* Silhouette score
* Davies–Bouldin score
* Calinski–Harabasz score

**Product purchasing patterns**

* Product combinations frequently purchased by the same customers

These outputs help translate raw purchase data into **actionable business insights**.

---

## Technologies Used

* **Python**
* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Scikit‑learn** – clustering algorithms
* **Matplotlib / Seaborn** – data visualization
* **SciPy** – distance calculations

---

## Project Structure

```
customer-segmentation-and-market-basket-analysis

├── data/
│   └── walmart.csv

├── outputs/
│   ├── plots/
│   └── tables/

├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── clustering.py
│   ├── market_basket.py
│   └── main.py

├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/GinoBiasioli/customer-segmentation-and-market-basket-analysis.git
```

Navigate to the project folder:

```bash
cd customer-segmentation-and-market-basket-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the full analysis pipeline:

```bash
python src/main.py
```

The pipeline performs:

1. Data loading
2. Data preprocessing
3. Cluster evaluation
4. Customer segmentation
5. Cluster profiling
6. Purchase pattern analysis

Generated plots and tables are saved in the **outputs/** folder.

---

## Author

**Gino Biasioli**

Data Science Master's Student
