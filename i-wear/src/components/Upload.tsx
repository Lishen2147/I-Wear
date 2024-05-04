"use client";

import React, { useState } from 'react';
import UploadForm from './subcomponents/Upload/UploadForm';

const Upload: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  return (
    <div>
      <UploadForm onFileChange={handleImageChange} image={image} />
    </div>
  );
};

export default Upload;