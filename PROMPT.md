# 🎯 URL Classification Prompt

This document showcases the prompt engineering approach used for URL security classification using GPT-4o-mini.

## 📋 System Message

```
You are a URL security expert. 
Evaluate the probability that the given URL is malicious or benign. 
Malicious includes (phishing, malware, defacement, or other harmful). 
Output ONLY a single JSON object with one key 'score' whose value is a number in [0,1]. 
No text before or after the JSON.
```

## 💬 User Prompt Template

```
You are a URL security expert. Evaluate the probability that the given URL is malicious or benign.

Malicious URLs often exhibit abnormal lexical or structural patterns—such as high-entropy or brand-mismatched domains, risky TLDs, suspicious keywords, executable or archive downloads, cloud-storage abuse, redirects/shorteners, and compromised CMS traces—while lacking the consistent branding, clean semantics, and trusted hosting typical of benign sites.

Benign URLs are known reputable sites with clean descriptive paths. Normal encoding alone is neutral.

Examples:
rapreviews.com/archive/2010_04_cityofdirt.html (benign)
Signal: Established review site; semantic and readable path; no suspicious parameters or brand terms. 

http://viva-sport.com/index.php?option=com_virtuemart&page=shop.browse&category_id=19&Itemid=54 (malicious)
Signal: HTTP (no TLS); Joomla/VirtueMart CMS stack with long query string; typical signs of defacement or compromise. 

http://fr57f6yg.gb.net/boa/login.php?... (malicious)
Signal: Subdomain appears high-entropy/random; path contains "boa/login.php" (brand keyword mismatch — brand impersonation); very long session parameters. 

Show me your thinking process, but only end your response with one-line JSON exactly like: {"score": <float>} (no extra text).

URL: {url}
Return:
```

## 🔍 Prompt Design Principles

### 1. **Role Definition**
- Establishes the AI as a "URL security expert"
- Sets clear expectations for the task

### 2. **Clear Classification Criteria**
- **Malicious indicators**: High-entropy domains, brand mismatches, risky TLDs, suspicious keywords, executable downloads, cloud-storage abuse, redirects/shorteners, compromised CMS
- **Benign indicators**: Reputable sites, clean semantic paths, trusted hosting

### 3. **Few-Shot Learning**
- Provides concrete examples with explanations
- Shows both benign and malicious cases
- Includes reasoning signals for each example

### 4. **Structured Output**
- Requires JSON format: `{"score": <float>}`
- Score range: [0, 1] where 0 = benign, 1 = malicious
- Encourages thinking process but constrains final output

### 5. **Chain of Thought**
- "Show me your thinking process" encourages reasoning
- Helps the model consider multiple factors before scoring

## 📊 Model Configuration

- **Model**: `gpt-4o-mini`
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 200
- **Response Format**: JSON object
- **Threshold**: 0.2 (scores ≥ 0.2 classified as malicious)

## 🎨 Prompt Engineering Techniques Used

1. **Role-based prompting**: Assigns expert identity
2. **Few-shot examples**: Provides learning examples
3. **Chain of thought**: Encourages reasoning
4. **Structured output**: JSON format for easy parsing
5. **Signal-based reasoning**: Highlights key indicators

## 📈 Performance

The prompt achieves good performance on URL classification tasks. See the confusion matrix and evaluation metrics in the evaluation results.

