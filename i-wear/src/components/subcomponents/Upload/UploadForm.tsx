"use client";

import React, { Dispatch, SetStateAction } from "react";

interface UploadFormProps {
  image: File | null;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const UploadForm: React.FC<UploadFormProps> = ({ image, onFileChange }) => {
  return (
    <div>
      <h2>Upload Photo</h2>
      <input type="file" accept="image/*" onChange={onFileChange} />
      {image && (
        <div>
          <img src={URL.createObjectURL(image)} alt="Uploaded" />
        </div>
      )}
    </div>
  );
};

export default UploadForm;
