"use client";

import React, { useState } from 'react';

const Upload: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  return (
    <div>
      <div>
        <h2>Upload Photo</h2>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        {image && (
          <div>
            <img src={URL.createObjectURL(image)} alt="Uploaded" />
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;