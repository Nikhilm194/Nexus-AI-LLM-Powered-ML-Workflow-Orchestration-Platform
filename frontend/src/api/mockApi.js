/**
 * Mock API service for pipeline generation.
 * Will be replaced with real FastAPI calls later.
 */

const MOCK_DELAY_MS = 1500;

/**
 * Simulates a POST to /api/generate-pipeline
 * @param {string} userIntent - The user's described intent
 * @param {File|null} csvFile - Optional uploaded CSV file
 * @returns {Promise<Array>} A 3-step pipeline array
 */
export async function generatePipeline(userIntent, csvFile) {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, MOCK_DELAY_MS));

  // Build a contextual pipeline based on keywords in the intent
  const intentLower = userIntent.toLowerCase();

  if (intentLower.includes('classify') || intentLower.includes('classification')) {
    return [
      {
        id: 'step-1',
        name: 'Data Ingestion',
        type: 'ingestion',
        description: 'Load and validate CSV dataset for classification',
        config: {
          source: csvFile ? csvFile.name : 'dataset.csv',
          validation: true,
          encoding: 'utf-8',
        },
        icon: 'database',
      },
      {
        id: 'step-2',
        name: 'Feature Engineering',
        type: 'transform',
        description: 'Scale features, encode categoricals, handle missing values',
        config: {
          scaler: 'StandardScaler',
          encoder: 'OneHotEncoder',
          imputer: 'KNNImputer',
        },
        icon: 'transform',
      },
      {
        id: 'step-3',
        name: 'Random Forest Classifier',
        type: 'model',
        description: 'Train and evaluate a Random Forest model',
        config: {
          algorithm: 'RandomForestClassifier',
          n_estimators: 200,
          max_depth: 15,
          cv_folds: 5,
        },
        icon: 'model',
      },
    ];
  }

  if (intentLower.includes('cluster') || intentLower.includes('segment')) {
    return [
      {
        id: 'step-1',
        name: 'Data Ingestion',
        type: 'ingestion',
        description: 'Ingest raw data and perform initial profiling',
        config: {
          source: csvFile ? csvFile.name : 'dataset.csv',
          profiling: true,
        },
        icon: 'database',
      },
      {
        id: 'step-2',
        name: 'Dimensionality Reduction',
        type: 'transform',
        description: 'Apply PCA to reduce feature space for clustering',
        config: {
          method: 'PCA',
          n_components: 3,
          normalize: true,
        },
        icon: 'transform',
      },
      {
        id: 'step-3',
        name: 'K-Means Clustering',
        type: 'model',
        description: 'Cluster data points using K-Means algorithm',
        config: {
          algorithm: 'KMeans',
          n_clusters: 5,
          max_iter: 300,
        },
        icon: 'model',
      },
    ];
  }

  // Default: regression pipeline
  return [
    {
      id: 'step-1',
      name: 'Data Ingestion',
      type: 'ingestion',
      description: 'Load, validate, and profile the input dataset',
      config: {
        source: csvFile ? csvFile.name : 'dataset.csv',
        validation: true,
        profiling: true,
      },
      icon: 'database',
    },
    {
      id: 'step-2',
      name: 'Preprocessing & Feature Engineering',
      type: 'transform',
      description: 'Clean data, engineer features, and normalize values',
      config: {
        scaler: 'MinMaxScaler',
        imputer: 'SimpleImputer',
        feature_selection: 'mutual_info_regression',
      },
      icon: 'transform',
    },
    {
      id: 'step-3',
      name: 'XGBoost Regressor',
      type: 'model',
      description: 'Train a gradient-boosted regression model',
      config: {
        algorithm: 'XGBRegressor',
        n_estimators: 300,
        learning_rate: 0.05,
        max_depth: 8,
      },
      icon: 'model',
    },
  ];
}
