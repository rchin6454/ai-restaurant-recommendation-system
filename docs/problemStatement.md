# Problem Statement

## AI-Powered Restaurant Recommendation System (Zomato Use Case)

You are tasked with building an **AI-powered restaurant recommendation service** inspired by Zomato. The system should intelligently suggest restaurants based on user preferences by combining structured data with a Large Language Model (LLM).

---

## Objective

Design and implement an application that:

- Takes user preferences (such as location, budget, cuisine, and ratings)
- Uses a real-world dataset of restaurants
- Leverages an LLM to generate personalized, human-like recommendations
- Displays clear and useful results to the user

---

## System Workflow

### 1. Data Ingestion

- Load and preprocess the Zomato dataset from Hugging Face
  (https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- Extract relevant fields such as restaurant name, location, cuisine, cost, rating, etc.

### 2. User Input

- Collect user preferences:
  - Location (e.g., Delhi, Bangalore)
  - Budget (low, medium, high)
  - Cuisine (e.g., Italian, Chinese)
  - Minimum rating
  - Any additional preferences (e.g., family-friendly, quick service)

### 3. Integration Layer

- Filter and prepare relevant restaurant data based on user input
- Pass structured results into an LLM prompt
- Design a prompt that helps the LLM reason and rank options

### 4. Recommendation Engine

- Use the LLM to:
  - Rank restaurants
  - Provide explanations (why each recommendation fits)
  - Optionally summarize choices

### 5. Output Display

- Present top recommendations in a user-friendly format:
  - Restaurant Name
  - Cuisine
  - Rating
  - Estimated Cost
  - AI-generated explanation

---

## Project Context

| Item | Value |
| --- | --- |
| Project name | AI-Powered Restaurant Recommendation System |
| Source document | `Problem Statement.rtf` |
| Dataset | `ManikaSaini/zomato-restaurant-recommendation` (Hugging Face) |
| Core idea | Structured filtering over restaurant data + LLM reasoning for ranking and explanation |

### Key Design Points

- **Hybrid approach:** deterministic filtering narrows the dataset; the LLM handles ranking, reasoning, and natural-language explanation. The LLM should not be asked to search the raw dataset on its own.
- **Prompt design matters:** the prompt must carry the filtered structured records plus the user's stated preferences, and instruct the model to rank and justify.
- **Explainability is a requirement, not a bonus:** every recommendation must come with a reason tied to the user's preferences.
- **Free-form preferences** (e.g., "family-friendly", "quick service") are handled by the LLM layer rather than by structured filters.
