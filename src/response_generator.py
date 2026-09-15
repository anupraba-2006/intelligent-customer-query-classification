
from .preprocessing import extract_order_id, extract_quantity

TEMPLATES = {
    "Quotation": (
        "Thank you for your interest{qty_clause}. Our sales team will share "
        "a detailed quotation shortly. If you need this urgently, please "
        "let us know your deadline."
    ),
    "Order Status": (
        "Thank you for reaching out{order_clause}. We're checking the latest "
        "status with our logistics team and will update you shortly."
    ),
    "Complaint": (
        "We're sorry to hear about this{order_clause}. This has been logged "
        "as a priority issue and our support team will contact you shortly "
        "to resolve it."
    ),
    "General Enquiry": (
        "Thanks for your question. Our team will get back to you with the "
        "details shortly. Feel free to share any additional information."
    ),
}

FALLBACK_TEMPLATE = (
    "Thank you for contacting us. Your query has been received and a "
    "member of our team will review it shortly."
)


def generate_response(text: str, category: str, human_review_flag: bool) -> str:
    order_id = extract_order_id(text)
    quantity = extract_quantity(text)

    qty_clause = f" in {quantity} units" if quantity else ""
    order_clause = f" regarding order {order_id}" if order_id else ""

    template = TEMPLATES.get(category, FALLBACK_TEMPLATE)
    response = template.format(qty_clause=qty_clause, order_clause=order_clause)

    if human_review_flag:
        response += (
            " (Note: this query has been flagged for review by our team to "
            "ensure you get the most accurate response.)"
        )
    return response