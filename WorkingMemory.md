# Working Memory

## **Phase 1: Strengthening the Foundations**  
### **1. Improve Vector and Matrix Operations**
- ✅ **Current:** Basic `Vector` and `Matrix` classes implemented with tests.  
- ✅ **Next Steps:**  
  - Optimize matrix operations (**dot product, transpose, determinant, inverse, row/column operations**). 
  - Implement **broadcasting** for element-wise operations.  
  - Implement **efficient memory management** (e.g., avoiding unnecessary copies).  
- 🔲 Update Test cases
- 🔲 Update README

### **2. Enhance the Perceptron**
- ✅ **Current:** Basic perceptron implemented with tests.  
- 🔲 **Next Steps:**  
  - Modify perceptron to accept **batch updates** for training efficiency.  
  - Generalize perceptron to support **multi-class classification** (One-vs-All or Softmax).  
- 🔲 Update Test cases
- 🔲 Update README

---

## **Phase 2: Expanding Classic Machine Learning Models**
### **3. Implementing Linear Regression**
- 🔲 Implement **Ordinary Least Squares (OLS)** regression.  
- 🔲 Implement **Gradient Descent-based Linear Regression** for large datasets.  
- 🔲 Update README

### **4. Implementing Logistic Regression**
- 🔲 Implement **Logistic Regression for binary classification**.  
- 🔲 Extend to **multi-class classification using Softmax Regression**.  
- 🔲 Implement **L1 and L2 regularization** (Lasso & Ridge Regression).  
- 🔲 Update README

### **5. Support Vector Machines (SVMs)**
- 🔲 Implement **hard-margin and soft-margin SVMs**.  
- 🔲 Implement **Quadratic Programming Solver** (or use an optimization technique like SMO).  
- 🔲 Implement **Kernel SVMs with RBF, polynomial, and linear kernels**.  
- 🔲 Update README

### **6. Implement Kernel PCA**
- 🔲 Implement **Principal Component Analysis (PCA)** for dimensionality reduction.  
- 🔲 Implement **Kernel PCA** with support for RBF, polynomial, and custom kernels.  
- 🔲 Update README
---

## **Phase 3: Neural Networks (MLPs) and Deep Learning**
### **7. Implementing a Multi-Layer Perceptron (MLP)**
- 🔲 Implement a **Multi-Layer Perceptron (MLP)** class with:  
  - Support for multiple layers (hidden layers).  
  - Configurable **activation functions**:  
    - Sigmoid, Tanh, ReLU, LeakyReLU, Softmax.  
  - Matrix-based forward propagation for efficiency.  
- 🔲 Update README

### **8. Backpropagation & Optimization**
- 🔲 Implement **backpropagation**:  
  - Compute gradients using **chain rule**.  
  - Update weights using **gradient descent**.  
- 🔲 Implement **learning rate schedules** (e.g., step decay, exponential decay).  
- 🔲 Add support for **SGD, Momentum, Adam, RMSprop** optimizers.  
- 🔲 Update README

### **9. Loss Functions**
- 🔲 Implement standard **loss functions**:  
  - Mean Squared Error (MSE) (for regression).  
  - Cross-Entropy Loss (for classification).  
  - Hinge Loss (for SVM-like models).  
- 🔲 Update README
---

## **Phase 4: Regularization and Model Generalization**
### **10. Regularization Techniques**
- 🔲 Implement **L1/L2 regularization (weight decay)**.  
- 🔲 Implement **dropout** to improve generalization.  
- 🔲 Update README
### **11. Model Evaluation & Metrics**
- 🔲 Implement **accuracy, precision, recall, F1-score, confusion matrix**.  
- 🔲 Support **validation & test splits** for training models properly.  
- 🔲 Update README
---

## **Phase 5: Expanding Model Capabilities**
### **12. Implementing Convolutional Neural Networks (CNNs)**
- 🔲 Implement **convolution layers** for image processing.  
- 🔲 Implement **max pooling & average pooling**.  
- 🔲 Implement **basic image preprocessing** (normalization, grayscale conversion).  
- 🔲 Update README

### **13. Implementing Recurrent Neural Networks (RNNs)**
- 🔲 Implement **basic RNN cell**.  
- 🔲 Implement **Long Short-Term Memory (LSTM) & Gated Recurrent Units (GRU)**.  
- 🔲 Update README
---

## **Phase 6: Performance & Efficiency**
### **14. Optimization & Parallelization**
- 🔲 Implement **automatic differentiation** for gradient computation.  
- 🔲 Optimize matrix computations using **parallelization (multi-threading or SIMD)**.  
- 🔲 Consider **GPU acceleration** with Rust libraries like `wgpu` or `cust` (CUDA bindings).  
- 🔲 Update README
---