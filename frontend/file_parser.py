def extract_text(file) -> str:
    """
    Extracts plain text from file-like UploadedFile objects based on extension.
    Supports .txt, .rtf, and .pdf.
    """
    file.seek(0)
    filename = file.name.lower()
    
    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="replace")
        
    elif filename.endswith(".rtf"):
        from striprtf.striprtf import rtf_to_text
        rtf_data = file.read().decode("utf-8", errors="ignore")
        return rtf_to_text(rtf_data)
        
    elif filename.endswith(".pdf"):
        import pdfplumber
        import re
        with pdfplumber.open(file) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Normalize line endings
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    
                    # Process line-by-line to reconstruct paragraphs
                    lines = [line.strip() for line in text.split("\n")]
                    chunks = []
                    current_chunk = []
                    
                    for line in lines:
                        if not line:
                            continue
                        
                        current_chunk.append(line)
                        
                        # Identify paragraph breaks:
                        # 1. Ends with punctuation (e.g., . ! ? : …) optionally followed by quotes/brackets
                        ends_with_punctuation = bool(re.search(r'[.!?…:]\s*["\')\]]*$', line))
                        # 2. Is it a short line (e.g. a heading, list item, or title)?
                        is_short = len(line) < 50
                        
                        if ends_with_punctuation or is_short:
                            # End of paragraph/heading: join lines with space and save
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                    
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    
                    # Join page paragraphs with double newlines
                    pages_text.append("\n\n".join(chunks))
                    
            return "\n\n".join(pages_text)
            
    else:
        raise ValueError(f"Unsupported file format: {file.name}")
