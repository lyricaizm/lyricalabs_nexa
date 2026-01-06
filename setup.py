from setuptools import setup, find_packages

setup(
    name="Lyrica_Labs_Nexa_LLM",
    version="0.3.0",
    packages=find_packages(),
    install_requires=["requests"],
    python_requires=">=3.8",
    description="Nexa: Lyrica Labs tarafından eğitilmiş geniş veri LLM Python kütüphanesi",
    url="https://lyricalabs.vercel.app/nexa",
    author="Lyrica Labs",
    author_email="lyricalabs@gmail.com",
    license="MIT"
)
