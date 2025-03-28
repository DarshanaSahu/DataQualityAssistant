from sqlalchemy import create_engine, text
from app.core.config import settings
import random
from datetime import datetime, timedelta

def seed_database():
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as connection:
        # Create authors table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS authors (
                author_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE,
                birth_date DATE,
                country VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # Create books table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS books (
                book_id SERIAL PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                author_id INTEGER REFERENCES authors(author_id),
                isbn VARCHAR(17) UNIQUE,
                publication_date DATE,
                price DECIMAL(10,2),
                genre VARCHAR(50),
                page_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # Create publishers table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS publishers (
                publisher_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                address TEXT,
                phone VARCHAR(20),
                email VARCHAR(100) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # Create book_publishers junction table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS book_publishers (
                book_id INTEGER REFERENCES books(book_id),
                publisher_id INTEGER REFERENCES publishers(publisher_id),
                published_date DATE,
                edition VARCHAR(20),
                PRIMARY KEY (book_id, publisher_id)
            );
        """))
        
        # Insert sample data
        # Authors
        authors_data = [
            ("John Smith", "john.smith@email.com", "1980-05-15", "USA"),
            ("Emma Wilson", "emma.wilson@email.com", "1990-08-22", "UK"),
            ("Michael Brown", "michael.brown@email.com", "1975-03-10", "Canada"),
            ("Sarah Davis", "sarah.davis@email.com", "1985-12-01", "Australia"),
            ("David Lee", "david.lee@email.com", "1995-07-30", "Singapore")
        ]
        
        for author in authors_data:
            connection.execute(text("""
                INSERT INTO authors (name, email, birth_date, country)
                VALUES (:name, :email, :birth_date, :country)
                ON CONFLICT (email) DO NOTHING
            """), {
                "name": author[0],
                "email": author[1],
                "birth_date": author[2],
                "country": author[3]
            })
        
        # Publishers
        publishers_data = [
            ("Penguin Books", "123 Penguin Street, London", "+44-20-7123-4567", "penguin@books.com"),
            ("Random House", "456 Random Ave, New York", "+1-212-555-0123", "random@house.com"),
            ("HarperCollins", "789 Harper Road, Toronto", "+1-416-555-0123", "harper@collins.com"),
            ("Simon & Schuster", "321 Simon Street, Sydney", "+61-2-5555-0123", "simon@schuster.com")
        ]
        
        for publisher in publishers_data:
            connection.execute(text("""
                INSERT INTO publishers (name, address, phone, email)
                VALUES (:name, :address, :phone, :email)
                ON CONFLICT (email) DO NOTHING
            """), {
                "name": publisher[0],
                "address": publisher[1],
                "phone": publisher[2],
                "email": publisher[3]
            })
        
        # Books
        books_data = [
            ("The Great Adventure", 1, "978-0-123456-47-2", "2020-01-15", 29.99, "Fiction", 350),
            ("Data Science Basics", 2, "978-0-123456-48-9", "2021-03-20", 49.99, "Non-Fiction", 450),
            ("Mystery of the Night", 3, "978-0-123456-49-6", "2019-11-10", 24.99, "Mystery", 300),
            ("Python Programming", 4, "978-0-123456-50-2", "2022-06-05", 39.99, "Technical", 400),
            ("The Art of Cooking", 5, "978-0-123456-51-9", "2021-09-15", 34.99, "Cookbook", 250)
        ]
        
        for book in books_data:
            connection.execute(text("""
                INSERT INTO books (title, author_id, isbn, publication_date, price, genre, page_count)
                VALUES (:title, :author_id, :isbn, :publication_date, :price, :genre, :page_count)
                ON CONFLICT (isbn) DO NOTHING
            """), {
                "title": book[0],
                "author_id": book[1],
                "isbn": book[2],
                "publication_date": book[3],
                "price": book[4],
                "genre": book[5],
                "page_count": book[6]
            })
        
        # Book Publishers
        book_publishers_data = [
            (1, 1, "2020-01-15", "1st Edition"),
            (2, 2, "2021-03-20", "1st Edition"),
            (3, 3, "2019-11-10", "1st Edition"),
            (4, 4, "2022-06-05", "2nd Edition"),
            (5, 1, "2021-09-15", "1st Edition")
        ]
        
        for bp in book_publishers_data:
            connection.execute(text("""
                INSERT INTO book_publishers (book_id, publisher_id, published_date, edition)
                VALUES (:book_id, :publisher_id, :published_date, :edition)
                ON CONFLICT (book_id, publisher_id) DO NOTHING
            """), {
                "book_id": bp[0],
                "publisher_id": bp[1],
                "published_date": bp[2],
                "edition": bp[3]
            })
        
        connection.commit()

if __name__ == "__main__":
    seed_database() 