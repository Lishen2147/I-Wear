"use client";
import Image from "next/image";

const POST_URL = "http://localhost:8080/predict";

export default function Home() {
  const formSubmitHandler = async (e: any) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    console.log(formData);

    const data = await fetch(POST_URL, {
      method: "POST",
      body: formData,
    });

    const response = await data.json(); // returns json of prediction & error

    console.log(response);
  };

  return (
    <form
      method="post"
      encType="multipart/form-data"
      onSubmit={formSubmitHandler}
    >
      <input
        type="file"
        name="uploadFile"
        accept="image/png, image/jpg, image/jpeg"
        required
      />
      <br />
      <br />
      <input type="submit" />
    </form>
  );
}
