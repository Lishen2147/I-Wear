"use client";

import React from "react";

interface UploadFormProps {
  image: string | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const UploadForm: React.FC<UploadFormProps> = ({ image, onFileChange }) => {
  return (
    <div>
      <h2>Upload Photo</h2>
      <input type="file" accept="image/*" onChange={onFileChange} />
      {image && (
        <div>
          <img src={image} alt="Uploaded" />
        </div>
      )}
    </div>
  );
};

export default UploadForm;
