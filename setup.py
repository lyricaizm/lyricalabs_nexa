from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements.txt dosyasını okumak yerine direkt liste kullan
requirements = ["requests"]

setup(
    name="lyricalabs_nexa",
    version="0.3.1",
    author="Lyrica Labs",
    author_email="lyricalabs@gmail.com",
    description="Lyrica Labs Nexa LLM API Python İstemci Kütüphanesi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyricaizm/lyricalabs_nexa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=["ai", "llm", "lyricalabs", "nexa", "api", "text-generation", "insomnia"],
    project_urls={
        "Documentation": "https://lyricalabs.vercel.app/docs",
        "Source": "https://github.com/lyricaizm/lyricalabs_nexa",
        "Tracker": "https://github.com/lyricaizm/lyricalabs_nexa/issues",
    },
)
