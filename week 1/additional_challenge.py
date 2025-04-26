books = [
    {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "genre": "Fiction",
        "rating": 4.2
    },
    {
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "genre": "Classic",
        "rating": 4.5
    },
    {
        "title": "1984",
        "author": "George Orwell",
        "genre": "Dystopian",
        "rating": 4.8
    },
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "genre": "Romance",
        "rating": 4.7
    },
    {
        "title": "Harry Potter and the Sorcerer's Stone",
        "author": "J.K. Rowling",
        "genre": "Fantasy",
        "rating": 4.9
    },
    {
        "title": "The Catcher in the Rye",
        "author": "J.D. Salinger",
        "genre": "Coming-of-age",
        "rating": 4.1
    }
]


# 1. Check Book Ratings
def check_rating(book):
    # if book.rating > 4.5:
    #     return True
    # return False
    rating = book["rating"]
    if rating <= 4.0:
        return "low"
    elif 4.0 < rating <= 4.5:
        return "medium"
    elif rating > 4.5:
        return "high"

print(check_rating(books[2]))


# 2. Average Rating by Genre
def average_rating_by_genre(books, genre):
    total_rating = 0.0
    count = 0
    for book in books:
        if book["genre"] == genre:
            count += 1
            total_rating += book["rating"]
    if count == 0:
        return "No such genre in books"
    return total_rating / count

print(average_rating_by_genre(books, "Fiction"))


# 3. Books by Author
def books_by_author(books, author):
    written_by_author = []
    for book in books:
        if book["author"] == author:
            written_by_author.append(book["title"])
    if len(written_by_author) == 0:
        return "The author is not in the list"
    return written_by_author

print(books_by_author(books, "Harper Lee"))
