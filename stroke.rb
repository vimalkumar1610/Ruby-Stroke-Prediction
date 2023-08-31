$VERBOSE = nil

require 'csv'
require 'liblinear'

x_data = []
y_data = []

# Load data from CSV file into two arrays - one for independent variables X and one for the dependent variable Y
CSV.foreach("stroke_data.csv", :headers => true) do |row|
  gender = row["gender"] == "Female" ? 1 : 0
  age = row["age"].to_f
  hypertension = row["hypertension"].to_i
  heart_disease = row["heart_disease"].to_i
  ever_married = row["ever_married"] == "Yes" ? 1 : 0
  work_type = row["work_type"].to_i  # You may need to encode this categorical feature appropriately
  residence_type = row["Residence_type"] == "Urban" ? 1 : 0
  avg_glucose_level = row["avg_glucose_level"].to_f
  bmi = row["bmi"].to_f
  smoking_status = row["smoking_status"].to_i  # You may need to encode this categorical feature appropriately
  
  # Prepare the feature vector
  x_data.push([gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status])
  
  # Prepare the target variable
  y_data.push(row["stroke"].to_i)
end

# Size of dataset
puts "Size of dataset: #{x_data.size}"

# Divide data into a training set and test set
test_size_percentage = 20.0
test_set_size = (x_data.size * (test_size_percentage / 100.0)).to_i 

len = y_data.size - test_set_size

# Size of training set
puts "Size of training data: #{len}"

# Size of test data
puts "Size of test data: #{test_set_size}"

test_x_data = x_data[len...y_data.size]
test_y_data = y_data[len...y_data.size]

training_x_data = x_data[0...len]
training_y_data = y_data[0...len]

# Setup model and train using training data
model = Liblinear.train(
  { solver_type: Liblinear::L2R_LR },   # Solver type: L2R_LR - L2-regularized logistic regression
  training_y_data,                      # Training data classification
  training_x_data,                      # Training data independent variables
  100                                   # Bias
)

# Predict class
prediction = Liblinear.predict(model, [0, 60, 0, 0, 1, 0, 1, 120, 25, 2])  # Example prediction for a test case

# Get prediction probabilities
probs = Liblinear.predict_probabilities(model, [0, 60, 0, 0, 1, 0, 1, 120, 25, 2])
probs = probs.sort

puts "Algorithm predicted class #{prediction}"
puts "#{(probs[1] * 100).round(2)}% probability of stroke"
puts "#{(probs[0] * 100).round(2)}% probability of no stroke"

predicted = []
test_x_data.each do |params|
  predicted.push(Liblinear.predict(model, params))
end

correct = predicted.each_with_index.count { |e, i| e == test_y_data[i] }

accuracy = (correct.to_f / test_set_size) * 100.0
puts "Accuracy: #{accuracy.round(2)}% - test set of size #{test_size_percentage}%"
