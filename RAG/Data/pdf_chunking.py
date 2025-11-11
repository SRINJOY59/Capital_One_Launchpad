import os
import fitz
from pathlib import Path
import shutil

class PDFChunker:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.pdf_dir = self.base_dir / "PDF"
        self.pdf_dir.mkdir(exist_ok=True)
        self.chunk_size = 30
    
    def clean_pdf_page(self, page):
        img_list = page.get_images()
        images_removed = 0
        
        for img_index, img in enumerate(img_list):
            xref = img[0]
            try:
                page.delete_image(xref)
                images_removed += 1
            except:
                pass
        
        if images_removed > 0:
            print(f"   🗑️ Removed {images_removed} images from page")
        
        annots = page.annots()
        for annot in annots:
            try:
                page.delete_annot(annot)
            except:
                pass
        
        return page
    
    def process_pdf(self, input_pdf_path):
        print(f"Processing PDF: {input_pdf_path}")
        
        try:
            source_doc = fitz.open(input_pdf_path)
            total_pages = len(source_doc)
            
            print(f"   Total pages: {total_pages}")
            print(f"   Creating chunks of {self.chunk_size} pages each")
            
            chunk_number = 1
            created_files = []
            
            for start_page in range(0, total_pages, self.chunk_size):
                end_page = min(start_page + self.chunk_size, total_pages)
                
                chunk_doc = fitz.open()
                
                print(f"   📝 Processing chunk {chunk_number}: pages {start_page + 1}-{end_page}")
                
                for page_num in range(start_page, end_page):
                    source_page = source_doc[page_num]
                    
                    cleaned_page = self.clean_pdf_page(source_page)
                    
                    chunk_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)
                
                input_name = Path(input_pdf_path).stem
                chunk_filename = f"{input_name}_chunk_{chunk_number:02d}.pdf"
                chunk_filepath = self.pdf_dir / chunk_filename
                
                chunk_doc.save(chunk_filepath)
                chunk_doc.close()
                
                created_files.append(chunk_filepath)
                print(f"   Saved: {chunk_filename}")
                
                chunk_number += 1
            
            source_doc.close()
            
            original_path = Path(input_pdf_path)
            if original_path.exists():
                original_path.unlink()
                print(f"    Removed original file: {original_path.name}")
            
            print(f"   Created {len(created_files)} chunks successfully!")
            return created_files
            
        except Exception as e:
            print(f"   ❌ Error processing PDF: {str(e)}")
            return []
    
    def process_all_pdfs_in_directory(self, source_dir=None):
        if source_dir is None:
            source_dir = self.base_dir
        
        search_paths = [
            Path(source_dir),
            Path(source_dir).parent,
            Path(source_dir).parent.parent,
            Path(source_dir) / "Dataset",
            Path(source_dir).parent / "Dataset",
            Path(source_dir).parent.parent / "Dataset",
            self.pdf_dir
        ]
        
        pdf_files = []
        
        for search_path in search_paths:
            if search_path.exists():
                found_pdfs = list(search_path.glob("*.pdf"))
                for pdf in found_pdfs:
                    if not '_chunk_' in pdf.name and pdf not in pdf_files:
                        pdf_files.append(pdf)
                        
        print(f"Searched in the following directories:")
        for path in search_paths:
            exists = "✓" if path.exists() else "✗"
            pdf_count = len(list(path.glob("*.pdf"))) if path.exists() else 0
            print(f"   {exists} {path} ({pdf_count} PDFs)")
        
        if not pdf_files:
            print("\nNo PDF files found to process")
            print("Place PDF files in any of the searched directories above")
            return
        
        print(f"\nFound {len(pdf_files)} PDF file(s) to process:")
        for pdf_file in pdf_files:
            print(f"   • {pdf_file}")
        print("=" * 60)
        
        total_chunks = 0
        
        for pdf_file in pdf_files:
            chunks = self.process_pdf(pdf_file)
            total_chunks += len(chunks)
            print("-" * 60)
        
        print(f"\nProcessing Summary:")
        print(f"   Files processed: {len(pdf_files)}")
        print(f"   Total chunks created: {total_chunks}")
        print(f"   Chunks saved in: {self.pdf_dir}")
        
        chunk_files = list(self.pdf_dir.glob("*.pdf"))
        if chunk_files:
            print(f"\nCreated chunk files:")
            for chunk_file in sorted(chunk_files):
                file_size = chunk_file.stat().st_size / 1024 / 1024
                print(f"   • {chunk_file.name} ({file_size:.1f} MB)")
    
    def get_pdf_info(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            info = {
                'pages': len(doc),
                'title': doc.metadata.get('title', 'Unknown'),
                'author': doc.metadata.get('author', 'Unknown'),
                'file_size_mb': Path(pdf_path).stat().st_size / 1024 / 1024
            }
            doc.close()
            return info
        except Exception as e:
            return {'error': str(e)}



def main():
    print("PDF Chunker - Extract 50-page chunks from large PDFs")
    print("Removes images and annotations while preserving text content")
    print("=" * 70)
    
    chunker = PDFChunker()
    
    chunker.process_all_pdfs_in_directory()
    
    print("\nPDF chunking process completed!")
    print(f"Check the '{chunker.pdf_dir}' directory for processed chunks")

if __name__ == "__main__":
    main()
