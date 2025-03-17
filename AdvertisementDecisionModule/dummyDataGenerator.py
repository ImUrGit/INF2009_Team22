import pandas as pd
import numpy as np

# Number of rows to generate
n = 10000

# Generate unique IDs from 1 to n
ids = np.arange(1, n + 1)

# Generate random ages between 18 and 65 (inclusive)
ages = np.random.randint(18, 66, size=n)

# Randomly assign genders ("M" or "F")
genders = np.random.choice(["M", "F"], size=n)

# Initialize counter column with 0 for all rows
counter = np.zeros(n, dtype=int)

# Define the 11 advertisement categories for grocery shoppers
categories = [
    "Fresh Produce",
    "Dairy & Eggs",
    "Meat & Seafood",
    "Bakery & Confectionery",
    "Pantry Staples",
    "Snacks & Beverages",
    "Organic & Health Foods",
    "Frozen & Ready Meals",
    "Household Essentials"
]

# Randomly assign a category to each row
category_column = np.random.choice(categories, size=n)

# Define keywords dictionary with at least 20 keywords per category
keywords_dict = {
    "Fresh Produce": "fruits, vegetables, organic, seasonal, apples, bananas, oranges, grapes, berries, lettuce, tomatoes, cucumbers, carrots, broccoli, spinach, kale, herbs, market, farm, fresh",
    "Dairy & Eggs": "milk, cheese, yogurt, eggs, butter, cream, ice cream, cottage cheese, curd, lactose-free, organic dairy, skim milk, whole milk, feta, mozzarella, cheddar, blue cheese, brie, gouda, dairy alternative",
    "Meat & Seafood": "beef, chicken, pork, lamb, turkey, salmon, tuna, shrimp, crab, lobster, bacon, sausage, grill, roast, steak, filet, organic meat, fresh fish, shellfish, marinated",
    "Bakery & Confectionery": "bread, pastries, cakes, desserts, cookies, brownies, muffins, croissants, bagels, donuts, pies, tarts, scones, buns, rolls, artisan, glazed, sweet, savory, bakery",
    "Pantry Staples": "pasta, rice, canned goods, spices, oil, vinegar, flour, sugar, salt, beans, lentils, broth, cereals, oats, baking soda, baking powder, condiments, noodles, grains, sauces",
    "Snacks & Beverages": "chips, soda, snacks, drinks, cookies, nuts, popcorn, juice, coffee, tea, energy drink, water, crackers, granola, bars, chocolate, iced tea, smoothies, biscuits, beverage",
    "Organic & Health Foods": "organic, gluten-free, vegan, healthy, superfoods, whole foods, natural, low-calorie, antioxidants, fiber, non-GMO, raw, plant-based, holistic, nutrition, sustainable, organic produce, free-range, herbal, health boost",
    "Frozen & Ready Meals": "frozen, quick, convenience, meals, ready-to-eat, microwave, frozen pizza, dinner, lunch, snacks, frozen veggies, frozen fruits, frozen dessert, ice cream, pre-cooked, frozen dinner, chilled, heat-and-serve, family meal, frozen food",
    "Household Essentials": "cleaning, supplies, paper, toiletries, detergent, dish soap, towels, napkins, tissues, batteries, light bulbs, trash bags, cleaners, household, maintenance, recycling, laundry, sponges, mops, essentials",
}

# Generate keywords column based on the assigned category for each row
keywords_column = [keywords_dict[cat] for cat in category_column]

# Create a DataFrame with the generated data
df = pd.DataFrame({
    "id": ids,
    "age": ages,
    "gender": genders,
    "counter": counter,
    "category": category_column,
    "keywords": keywords_column
})

# Save the DataFrame to a CSV file without the index column
df.to_csv("data.csv", index=False)

print("CSV file 'data.csv' with 10,000 rows has been created.")
